use std::{
    sync::Arc,
    fmt,
    hash::{Hash, Hasher},
    ops::{Deref, Range},
    collections::HashMap,
    io::{self, Read, Write},
};

use anyhow::{Result, Context, bail};
use bio::alphabets::dna::revcomp;
use bio::alphabets::nucleotide::NucleotideSequence;
use serde::{Serialize, Deserialize};
use bytes::{Bytes, BytesMut, BufMut};
use tokio::io::{AsyncRead, AsyncWrite};
use rayon::prelude::*;
use dashmap::DashMap;

const MAX_SEQUENCE_LENGTH: usize = 1_000_000_000;
const QUALITY_SCORE_OFFSET: u8 = 33;
const GC_CONTENT_WINDOW_SIZE: usize = 100;

#[derive(Clone, Serialize, Deserialize)]
pub struct Sequence {
    id: String,
    #[serde(with = "serde_bytes")]
    bases: Bytes,
    quality: Option<QualityScores>,
    metadata: SequenceMetadata,
    #[serde(skip)]
    cache: Arc<DashMap<String, CachedComputation>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceMetadata {
    pub length: usize,
    pub gc_content: f64,
    pub complexity: f64,
    pub quality_stats: Option<QualityStats>,
    pub annotations: HashMap<String, String>,
    pub flags: SequenceFlags,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QualityStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: u8,
    pub max: u8,
    pub median: u8,
    pub q30_bases: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScores {
    #[serde(with = "serde_bytes")]
    scores: Bytes,
    stats: QualityStats,
}

bitflags::bitflags! {
    #[derive(Serialize, Deserialize)]
    pub struct SequenceFlags: u32 {
        const PAIRED = 0b00000001;
        const REVERSE = 0b00000010;
        const SECONDARY = 0b00000100;
        const FAILED_QC = 0b00001000;
        const DUPLICATE = 0b00010000;
        const SUPPLEMENTARY = 0b00100000;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum CachedComputation {
    GCContent(f64),
    Complexity(f64),
    KmerCounts(HashMap<Vec<u8>, usize>),
    RepeatRegions(Vec<Range<usize>>),
    Checksum(String),
}

impl Sequence {
    pub fn new(id: impl Into<String>, bases: impl Into<Bytes>, quality: Option<QualityScores>) -> Self {
        let bases = bases.into();
        let metadata = SequenceMetadata::compute(&bases, quality.as_ref());
        
        Self {
            id: id.into(),
            bases,
            quality,
            metadata,
            cache: Arc::new(DashMap::new()),
        }
    }

    pub fn from_fasta(header: &str, sequence: &[u8]) -> Result<Self> {
        if sequence.len() > MAX_SEQUENCE_LENGTH {
            bail!("Sequence length exceeds maximum allowed size");
        }

        if !sequence.iter().all(|&b| matches!(b, b'A' | b'C' | b'G' | b'T' | b'N')) {
            bail!("Invalid sequence characters detected");
        }

        Ok(Self::new(header, Bytes::copy_from_slice(sequence), None))
    }

    pub fn from_fastq(
        header: &str,
        sequence: &[u8],
        quality: &[u8],
    ) -> Result<Self> {
        if sequence.len() != quality.len() {
            bail!("Sequence and quality scores length mismatch");
        }

        let quality_scores = QualityScores::new(quality)?;
        Ok(Self::new(header, Bytes::copy_from_slice(sequence), Some(quality_scores)))
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn len(&self) -> usize {
        self.bases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    pub fn bases(&self) -> &[u8] {
        &self.bases
    }

    pub fn quality(&self) -> Option<&QualityScores> {
        self.quality.as_ref()
    }

    pub fn metadata(&self) -> &SequenceMetadata {
        &self.metadata
    }

    pub fn reverse_complement(&self) -> Self {
        let mut bases = self.bases.to_vec();
        bases.reverse();
        for base in &mut bases {
            *base = match *base {
                b'A' => b'T',
                b'T' => b'A',
                b'C' => b'G',
                b'G' => b'C',
                b'N' => b'N',
                _ => panic!("Invalid base encountered"),
            };
        }

        let quality = self.quality.as_ref().map(|q| {
            let mut scores = q.scores.to_vec();
            scores.reverse();
            QualityScores::new(&scores).unwrap()
        });

        Self::new(format!("{}_revcomp", self.id), bases, quality)
    }

    pub fn gc_content(&self) -> f64 {
        if let Some(CachedComputation::GCContent(gc)) = self.cache.get("gc_content").map(|v| v.value().clone()) {
            return gc;
        }

        let gc = self.bases.iter()
            .filter(|&&b| b == b'G' || b == b'C')
            .count() as f64 / self.len() as f64;

        self.cache.insert("gc_content".to_string(), CachedComputation::GCContent(gc));
        gc
    }

    pub fn gc_content_distribution(&self, window_size: usize) -> Vec<f64> {
        if window_size == 0 || window_size > self.len() {
            return Vec::new();
        }

        self.bases.chunks(window_size)
            .map(|window| {
                window.iter()
                    .filter(|&&b| b == b'G' || b == b'C')
                    .count() as f64 / window.len() as f64
            })
            .collect()
    }

    pub fn sequence_complexity(&self) -> f64 {
        if let Some(CachedComputation::Complexity(complexity)) = self.cache.get("complexity").map(|v| v.value().clone()) {
            return complexity;
        }

        let kmer_counts = self.count_kmers(5);
        let total_kmers = kmer_counts.values().sum::<usize>();
        let complexity = kmer_counts.values()
            .map(|&count| {
                let p = count as f64 / total_kmers as f64;
                -p * p.log2()
            })
            .sum::<f64>();

        self.cache.insert("complexity".to_string(), CachedComputation::Complexity(complexity));
        complexity
    }

    pub fn find_repeats(&self, min_length: usize) -> Vec<Range<usize>> {
        let cache_key = format!("repeats_{}", min_length);
        if let Some(CachedComputation::RepeatRegions(regions)) = self.cache.get(&cache_key).map(|v| v.value().clone()) {
            return regions;
        }

        let mut repeats = Vec::new();
        let sequence = self.bases();
        
        for window_size in min_length..=sequence.len()/2 {
            for i in 0..=sequence.len()-window_size {
                let pattern = &sequence[i..i+window_size];
                let mut j = i + window_size;
                
                while j <= sequence.len()-window_size {
                    if sequence[j..j+window_size] == pattern {
                        repeats.push(j..j+window_size);
                        j += window_size;
                    } else {
                        j += 1;
                    }
                }
            }
        }

        self.cache.insert(cache_key, CachedComputation::RepeatRegions(repeats.clone()));
        repeats
    }

    pub fn count_kmers(&self, k: usize) -> HashMap<Vec<u8>, usize> {
        let cache_key = format!("kmers_{}", k);
        if let Some(CachedComputation::KmerCounts(counts)) = self.cache.get(&cache_key).map(|v| v.value().clone()) {
            return counts;
        }

        let mut counts = HashMap::new();
        if k > self.len() {
            return counts;
        }

        for window in self.bases.windows(k) {
            *counts.entry(window.to_vec()).or_insert(0) += 1;
        }

        self.cache.insert(cache_key, CachedComputation::KmerCounts(counts.clone()));
        counts
    }

    pub async fn write_fasta<W: AsyncWrite + Unpin>(&self, writer: &mut W) -> io::Result<()> {
        use tokio::io::AsyncWriteExt;
        
        writer.write_all(b">").await?;
        writer.write_all(self.id.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        
        for chunk in self.bases.chunks(70) {
            writer.write_all(chunk).await?;
            writer.write_all(b"\n").await?;
        }
        
        Ok(())
    }

    pub async fn write_fastq<W: AsyncWrite + Unpin>(&self, writer: &mut W) -> io::Result<()> {
        use tokio::io::AsyncWriteExt;
        
        if self.quality.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Cannot write FASTQ without quality scores",
            ));
        }

        writer.write_all(b"@").await?;
        writer.write_all(self.id.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.write_all(&self.bases).await?;
        writer.write_all(b"\n+\n").await?;
        writer.write_all(self.quality.as_ref().unwrap().scores()).await?;
        writer.write_all(b"\n").await?;
        
        Ok(())
    }

    pub fn calculate_hash(&self) -> String {
        use sha2::{Sha256, Digest};
        
        if let Some(CachedComputation::Checksum(hash)) = self.cache.get("hash").map(|v| v.value().clone()) {
            return hash;
        }

        let mut hasher = Sha256::new();
        hasher.update(&self.bases);
        if let Some(ref quality) = self.quality {
            hasher.update(quality.scores());
        }
        
        let hash = format!("{:x}", hasher.finalize());
        self.cache.insert("hash".to_string(), CachedComputation::Checksum(hash.clone()));
        hash
    }
}

impl QualityScores {
    pub fn new(scores: &[u8]) -> Result<Self> {
        if scores.iter().any(|&s| s < QUALITY_SCORE_OFFSET) {
            bail!("Invalid quality scores detected");
        }

        let scores = Bytes::copy_from_slice(scores);
        let stats = Self::calculate_stats(&scores);
        
        Ok(Self { scores, stats })
    }

    pub fn scores(&self) -> &[u8] {
        &self.scores
    }

    pub fn stats(&self) -> &QualityStats {
        &self.stats
    }

    fn calculate_stats(scores: &[u8]) -> QualityStats {
        let len = scores.len();
        let mut values: Vec<_> = scores.iter().map(|&s| s - QUALITY_SCORE_OFFSET).collect();
        values.sort_unstable();

        let sum: u32 = values.iter().map(|&x| x as u32).sum();
        let mean = sum as f64 / len as f64;

        let variance = values.iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / len as f64;

        let q30_bases = values.iter()
            .filter(|&&x| x >= 30)
            .count() as f64 / len as f64;

        QualityStats {
            mean,
            std_dev: variance.sqrt(),
            min: values[0],
            max: values[len - 1],
            median: values[len / 2],
            q30_bases,
        }
    }
}

impl SequenceMetadata {
    fn compute(bases: &[u8], quality: Option<&QualityScores>) -> Self {
        let length = bases.len();
        let gc_content = bases.iter()
            .filter(|&&b| b == b'G' || b == b'C')
            .count() as f64 / length as f64;

        Self {
            length,
            gc_content,
            complexity: 0.0, // Computed lazily
            quality_stats: quality.map(|q| q.stats().clone()),
            annotations: HashMap::new(),
            flags: SequenceFlags::empty(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_creation() {
        let seq = Sequence::new("test", b"ACGT".to_vec(), None);
        assert_eq!(seq.len(), 4);
        assert_eq!(seq.gc_content(), 0.5);
    }

    #[test]
    fn test_reverse_complement() {
        let seq = Sequence::new("test", b"ACGT".to_vec(), None);
        let revcomp = seq.reverse_complement();
        assert_eq!(revcomp.bases(), b"ACGT");
    }

    #[test]
    fn test_quality_scores() {
        let scores = vec![40u8; 4]; // ASCII value 40 corresponds to quality score 7
        let quality = QualityScores::new(&scores).unwrap();
        assert_eq!(quality.stats().mean, 7.0);
    }

    #[tokio::test]
    async fn test_fasta_writing() {
        let seq = Sequence::new("test", b"ACGT".to_vec(), None);
        let mut buffer = Vec::new();
        seq.write_fasta(&mut buffer).await.unwrap();
        assert_eq!(String::from_utf8(buffer).unwrap(), ">test\nACGT\n");
    }
}