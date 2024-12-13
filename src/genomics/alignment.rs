use std::{
    cmp::{max, min},
    collections::{HashMap, BTreeMap},
    sync::Arc,
    time::Instant,
};

use anyhow::{Result, Context, bail};
use bio::alignment::{pairwise, distance};
use bio::alphabets::dna::revcomp;
use ndarray::{Array2, Axis};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;

const DEFAULT_MATCH_SCORE: i32 = 2;
const DEFAULT_MISMATCH_PENALTY: i32 = -1;
const DEFAULT_GAP_OPEN_PENALTY: i32 = -5;
const DEFAULT_GAP_EXTEND_PENALTY: i32 = -1;
const MIN_ALIGNMENT_SCORE: i32 = 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    pub score: i32,
    pub alignment_type: AlignmentType,
    pub aligned_sequences: AlignedSequences,
    pub alignment_statistics: AlignmentStatistics,
    pub quality_metrics: AlignmentQuality,
    pub metadata: AlignmentMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedSequences {
    pub reference: Vec<u8>,
    pub query: Vec<u8>,
    pub alignment: Vec<u8>,
    pub start_position: usize,
    pub end_position: usize,
    pub gaps: Vec<GapLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentStatistics {
    pub identity: f64,
    pub similarity: f64,
    pub gaps: usize,
    pub mismatches: usize,
    pub alignment_length: usize,
    pub coverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentQuality {
    pub mapping_quality: u8,
    pub confidence_score: f64,
    pub error_probability: f64,
    pub local_quality_scores: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentMetadata {
    pub algorithm: String,
    pub parameters: AlignmentParameters,
    pub execution_time: f64,
    pub memory_usage: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AlignmentType {
    Global,
    Local,
    SemiGlobal,
    Custom(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentParameters {
    pub match_score: i32,
    pub mismatch_penalty: i32,
    pub gap_open_penalty: i32,
    pub gap_extend_penalty: i32,
    pub min_score: i32,
    pub max_gaps: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapLocation {
    pub position: usize,
    pub length: usize,
    pub sequence_type: SequenceType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SequenceType {
    Reference,
    Query,
}

pub struct Aligner {
    parameters: AlignmentParameters,
    algorithm_type: AlignmentType,
    score_matrix: Arc<ScoreMatrix>,
    cache: Arc<RwLock<AlignmentCache>>,
}

impl Aligner {
    pub fn new(parameters: AlignmentParameters, algorithm_type: AlignmentType) -> Self {
        let score_matrix = Arc::new(ScoreMatrix::new(
            parameters.match_score,
            parameters.mismatch_penalty,
        ));

        Self {
            parameters,
            algorithm_type,
            score_matrix,
            cache: Arc::new(RwLock::new(AlignmentCache::new())),
        }
    }

    pub async fn align(&self, query: &[u8], reference: &[u8]) -> Result<AlignmentResult> {
        let start_time = Instant::now();
        let cache_key = self.generate_cache_key(query, reference);

        if let Some(cached_result) = self.cache.read().await.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        let result = match self.algorithm_type {
            AlignmentType::Global => self.global_alignment(query, reference)?,
            AlignmentType::Local => self.local_alignment(query, reference)?,
            AlignmentType::SemiGlobal => self.semi_global_alignment(query, reference)?,
            AlignmentType::Custom(variant) => self.custom_alignment(query, reference, variant)?,
        };

        let execution_time = start_time.elapsed().as_secs_f64();
        let result = self.enrich_alignment_result(result, execution_time)?;

        self.cache.write().await.insert(cache_key, result.clone());
        Ok(result)
    }

    fn global_alignment(&self, query: &[u8], reference: &[u8]) -> Result<AlignmentResult> {
        let (score_matrix, traceback_matrix) = self.create_score_matrices(query, reference);
        
        let alignment = self.traceback_global(
            &score_matrix, 
            &traceback_matrix,
            query, 
            reference,
        )?;

        self.create_alignment_result(alignment, score_matrix, AlignmentType::Global)
    }

    fn local_alignment(&self, query: &[u8], reference: &[u8]) -> Result<AlignmentResult> {
        let mut score_matrix = Array2::zeros((query.len() + 1, reference.len() + 1));
        let mut traceback_matrix = Array2::zeros((query.len() + 1, reference.len() + 1));

        let mut max_score = 0;
        let mut max_position = (0, 0);

        for i in 1..=query.len() {
            for j in 1..=reference.len() {
                let (score, trace) = self.compute_local_cell_score(
                    &score_matrix,
                    i,
                    j,
                    query[i-1],
                    reference[j-1],
                );

                score_matrix[(i, j)] = score;
                traceback_matrix[(i, j)] = trace;

                if score > max_score {
                    max_score = score;
                    max_position = (i, j);
                }
            }
        }

        let alignment = self.traceback_local(
            &score_matrix,
            &traceback_matrix,
            query,
            reference,
            max_position,
        )?;

        self.create_alignment_result(alignment, score_matrix, AlignmentType::Local)
    }

    fn semi_global_alignment(&self, query: &[u8], reference: &[u8]) -> Result<AlignmentResult> {
        let mut score_matrix = Array2::zeros((query.len() + 1, reference.len() + 1));
        let mut traceback_matrix = Array2::zeros((query.len() + 1, reference.len() + 1));

        // Initialize first row with zeros (no penalty for starting gaps in reference)
        for i in 1..=query.len() {
            score_matrix[(i, 0)] = self.parameters.gap_open_penalty +
                (i as i32 - 1) * self.parameters.gap_extend_penalty;
        }

        for i in 1..=query.len() {
            for j in 1..=reference.len() {
                let (score, trace) = self.compute_semi_global_cell_score(
                    &score_matrix,
                    i,
                    j,
                    query[i-1],
                    reference[j-1],
                );

                score_matrix[(i, j)] = score;
                traceback_matrix[(i, j)] = trace;
            }
        }

        let max_last_row = score_matrix
            .row(query.len())
            .iter()
            .enumerate()
            .max_by_key(|&(_, &score)| score)
            .unwrap();

        let alignment = self.traceback_semi_global(
            &score_matrix,
            &traceback_matrix,
            query,
            reference,
            (query.len(), max_last_row.0),
        )?;

        self.create_alignment_result(alignment, score_matrix, AlignmentType::SemiGlobal)
    }

    fn custom_alignment(&self, query: &[u8], reference: &[u8], variant: u32) -> Result<AlignmentResult> {
        match variant {
            0 => self.banded_alignment(query, reference),
            1 => self.affine_gap_alignment(query, reference),
            2 => self.hybrid_alignment(query, reference),
            _ => bail!("Unsupported custom alignment variant"),
        }
    }

    fn compute_alignment_statistics(
        &self,
        aligned_sequences: &AlignedSequences,
    ) -> AlignmentStatistics {
        let mut identity = 0;
        let mut similarity = 0;
        let mut gaps = 0;
        let mut mismatches = 0;

        for ((&q, &r), &a) in aligned_sequences.query.iter()
            .zip(aligned_sequences.reference.iter())
            .zip(aligned_sequences.alignment.iter()) {
            if a == b'-' {
                gaps += 1;
            } else if q == r {
                identity += 1;
            } else {
                mismatches += 1;
                if self.score_matrix.is_similar(q, r) {
                    similarity += 1;
                }
            }
        }

        let alignment_length = aligned_sequences.alignment.len();
        let coverage = (alignment_length - gaps) as f64 / alignment_length as f64;

        AlignmentStatistics {
            identity: identity as f64 / alignment_length as f64,
            similarity: (identity + similarity) as f64 / alignment_length as f64,
            gaps,
            mismatches,
            alignment_length,
            coverage,
        }
    }

    fn compute_quality_metrics(
        &self,
        aligned_sequences: &AlignedSequences,
        score: i32,
    ) -> AlignmentQuality {
        let mapping_quality = self.calculate_mapping_quality(score);
        let confidence_score = self.calculate_confidence_score(aligned_sequences);
        let error_probability = (-confidence_score / 10.0).exp();
        let local_quality_scores = self.calculate_local_quality_scores(aligned_sequences);

        AlignmentQuality {
            mapping_quality,
            confidence_score,
            error_probability,
            local_quality_scores,
        }
    }

    fn calculate_mapping_quality(&self, score: i32) -> u8 {
        let scaled_score = (score as f64 / self.parameters.match_score as f64).min(254.0);
        scaled_score.round() as u8
    }

    fn calculate_confidence_score(&self, aligned_sequences: &AlignedSequences) -> f64 {
        let matches = aligned_sequences.query.iter()
            .zip(aligned_sequences.reference.iter())
            .filter(|(&q, &r)| q == r)
            .count();

        -10.0 * (1.0 - matches as f64 / aligned_sequences.alignment.len() as f64).log10()
    }

    fn calculate_local_quality_scores(&self, aligned_sequences: &AlignedSequences) -> Vec<f64> {
        aligned_sequences.query.iter()
            .zip(aligned_sequences.reference.iter())
            .map(|(&q, &r)| {
                if q == r {
                    1.0
                } else if self.score_matrix.is_similar(q, r) {
                    0.5
                } else {
                    0.0
                }
            })
            .collect()
    }
}

struct ScoreMatrix {
    matrix: Array2<i32>,
    similar_bases: HashMap<u8, Vec<u8>>,
}

impl ScoreMatrix {
    fn new(match_score: i32, mismatch_penalty: i32) -> Self {
        let mut matrix = Array2::zeros((256, 256));
        let mut similar_bases = HashMap::new();

        // Initialize standard DNA bases
        for &base in &[b'A', b'C', b'G', b'T', b'N'] {
            for &other in &[b'A', b'C', b'G', b'T', b'N'] {
                matrix[(base as usize, other as usize)] = if base == other {
                    match_score
                } else {
                    mismatch_penalty
                };
            }
        }

        // Define similar bases
        similar_bases.insert(b'A', vec![b'T']);
        similar_bases.insert(b'T', vec![b'A']);
        similar_bases.insert(b'C', vec![b'G']);
        similar_bases.insert(b'G', vec![b'C']);

        Self {
            matrix,
            similar_bases,
        }
    }

    fn score(&self, a: u8, b: u8) -> i32 {
        self.matrix[(a as usize, b as usize)]
    }

    fn is_similar(&self, a: u8, b: u8) -> bool {
        self.similar_bases
            .get(&a)
            .map_or(false, |similar| similar.contains(&b))
    }
}

#[derive(Debug)]
struct AlignmentCache {
    cache: HashMap<String, AlignmentResult>,
    max_size: usize,
}

impl AlignmentCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_size: 1000,
        }
    }

    fn get(&self, key: &str) -> Option<AlignmentResult> {
        self.cache.get(key).cloned()
    }

    fn insert(&mut self, key: String, value: AlignmentResult) {
        if self.cache.len() >= self.max_size {
            if let Some(oldest) = self.cache.keys().next().cloned() {
                self.cache.remove(&oldest);
            }
        }
        self.cache.insert(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn test_global_alignment() {
        let parameters = AlignmentParameters {
            match_score: DEFAULT_MATCH_SCORE,
            mismatch_penalty: DEFAULT_MISMATCH_PENALTY,
            gap_open_penalty: DEFAULT_GAP_OPEN_PENALTY,
            gap_extend_penalty: DEFAULT_GAP_EXTEND_PENALTY,
            min_score: MIN_ALIGNMENT_SCORE,
            max_gaps: None,
        };

        let aligner = Aligner::new(parameters, AlignmentType::Global);
        let query = b"ACGTACGT";
        let reference = b"ACGTACGT";

        let result = aligner.align(query, reference).await.unwrap();
        assert_eq!(result.alignment_statistics.identity, 1.0);
        assert_eq!(result.alignment_statistics.gaps, 0);
        assert_eq!(result.score, DEFAULT_MATCH_SCORE * 8);
    }

    #[tokio::test]
    async fn test_local_alignment() {
        let parameters = AlignmentParameters::default();
        let aligner = Aligner::new(parameters, AlignmentType::Local);
        let query = b"ACGTAAAACGT";
        let reference = b"ACGTACGT";

        let result = aligner.align(query, reference).await.unwrap();
        assert!(result.alignment_statistics.identity > 0.8);
        assert_eq!(result.alignment_type, AlignmentType::Local);
    }

    #[tokio::test]
    async fn test_semi_global_alignment() {
        let parameters = AlignmentParameters::default();
        let aligner = Aligner::new(parameters, AlignmentType::SemiGlobal);
        let query = b"TACGT";
        let reference = b"ACGTACGT";

        let result = aligner.align(query, reference).await.unwrap();
        assert!(result.score > 0);
        assert_eq!(result.alignment_type, AlignmentType::SemiGlobal);
    }

    #[tokio::test]
    async fn test_alignment_with_gaps() {
        let parameters = AlignmentParameters {
            gap_open_penalty: -2,
            gap_extend_penalty: -1,
            ..AlignmentParameters::default()
        };
        let aligner = Aligner::new(parameters, AlignmentType::Global);
        let query = b"ACGTGT";
        let reference = b"ACGTACGT";

        let result = aligner.align(query, reference).await.unwrap();
        assert!(result.alignment_statistics.gaps > 0);
        assert!(result.score < DEFAULT_MATCH_SCORE * query.len() as i32);
    }

    #[tokio::test]
    async fn test_alignment_quality_metrics() {
        let parameters = AlignmentParameters::default();
        let aligner = Aligner::new(parameters, AlignmentType::Global);
        let query = b"ACGTACGT";
        let reference = b"ACGTACGT";

        let result = aligner.align(query, reference).await.unwrap();
        assert!(result.quality_metrics.mapping_quality > 0);
        assert!(result.quality_metrics.confidence_score > 0.0);
        assert!(result.quality_metrics.error_probability < 0.01);
    }

    #[tokio::test]
    async fn test_alignment_cache() {
        let parameters = AlignmentParameters::default();
        let aligner = Aligner::new(parameters, AlignmentType::Global);
        let query = b"ACGTACGT";
        let reference = b"ACGTACGT";

        // First alignment should compute
        let first_result = aligner.align(query, reference).await.unwrap();
        let start_time = Instant::now();
        
        // Second alignment should use cache
        let second_result = aligner.align(query, reference).await.unwrap();
        let cached_time = start_time.elapsed();

        assert_eq!(first_result.score, second_result.score);
        assert!(cached_time.as_micros() < 1000); // Cache lookup should be fast
    }

    #[tokio::test]
    async fn test_custom_alignment_variants() {
        let parameters = AlignmentParameters::default();
        let aligner = Aligner::new(parameters, AlignmentType::Custom(0));
        let query = b"ACGTACGT";
        let reference = b"ACGTACGT";

        let result = aligner.align(query, reference).await.unwrap();
        assert!(result.score > 0);
    }

    #[tokio::test]
    async fn test_alignment_statistics() {
        let parameters = AlignmentParameters::default();
        let aligner = Aligner::new(parameters, AlignmentType::Global);
        let query = b"ACGTACGT";
        let reference = b"ACGTACGA";

        let result = aligner.align(query, reference).await.unwrap();
        assert!(result.alignment_statistics.identity < 1.0);
        assert!(result.alignment_statistics.identity > 0.8);
        assert_eq!(result.alignment_statistics.mismatches, 1);
    }

    #[tokio::test]
    async fn test_invalid_sequences() {
        let parameters = AlignmentParameters::default();
        let aligner = Aligner::new(parameters, AlignmentType::Global);
        let query = b"ACGTACGT";
        let empty_reference: &[u8] = b"";

        let result = aligner.align(query, empty_reference).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_score_matrix() {
        let score_matrix = ScoreMatrix::new(DEFAULT_MATCH_SCORE, DEFAULT_MISMATCH_PENALTY);
        
        assert_eq!(score_matrix.score(b'A', b'A'), DEFAULT_MATCH_SCORE);
        assert_eq!(score_matrix.score(b'A', b'G'), DEFAULT_MISMATCH_PENALTY);
        assert!(score_matrix.is_similar(b'A', b'T'));
        assert!(!score_matrix.is_similar(b'A', b'G'));
    }

    #[test]
    fn test_quality_metrics_calculation() {
        let aligned_sequences = AlignedSequences {
            reference: b"ACGTACGT".to_vec(),
            query: b"ACGTACGT".to_vec(),
            alignment: b"||||||||".to_vec(),
            start_position: 0,
            end_position: 8,
            gaps: vec![],
        };

        let parameters = AlignmentParameters::default();
        let aligner = Aligner::new(parameters, AlignmentType::Global);
        
        let quality = aligner.compute_quality_metrics(&aligned_sequences, DEFAULT_MATCH_SCORE * 8);
        assert!(quality.mapping_quality > 0);
        assert!(quality.confidence_score > 0.0);
        assert!(quality.error_probability < 1.0);
        assert_eq!(quality.local_quality_scores.len(), 8);
    }

    #[test]
    fn test_alignment_cache_behavior() {
        let mut cache = AlignmentCache::new();
        let result = AlignmentResult {
            score: 10,
            alignment_type: AlignmentType::Global,
            aligned_sequences: AlignedSequences {
                reference: vec![],
                query: vec![],
                alignment: vec![],
                start_position: 0,
                end_position: 0,
                gaps: vec![],
            },
            alignment_statistics: AlignmentStatistics {
                identity: 1.0,
                similarity: 1.0,
                gaps: 0,
                mismatches: 0,
                alignment_length: 0,
                coverage: 1.0,
            },
            quality_metrics: AlignmentQuality {
                mapping_quality: 0,
                confidence_score: 0.0,
                error_probability: 0.0,
                local_quality_scores: vec![],
            },
            metadata: AlignmentMetadata {
                algorithm: "test".to_string(),
                parameters: AlignmentParameters::default(),
                execution_time: 0.0,
                memory_usage: 0,
            },
        };

        // Test insertion
        cache.insert("test_key".to_string(), result.clone());
        assert!(cache.get("test_key").is_some());
        assert!(cache.get("nonexistent_key").is_none());

        // Test cache size limit
        for i in 0..2000 {
            cache.insert(format!("key_{}", i), result.clone());
        }
        assert!(cache.cache.len() <= cache.max_size);
    }

    proptest::proptest! {
        #[test]
        fn test_random_sequence_alignment(
            query in r"[ACGT]{1,100}",
            reference in r"[ACGT]{1,100}"
        ) {
            let parameters = AlignmentParameters::default();
            let aligner = Aligner::new(parameters, AlignmentType::Global);
            
            let result = tokio_test::block_on(aligner.align(
                query.as_bytes(),
                reference.as_bytes()
            ));
            
            prop_assert!(result.is_ok());
            let alignment = result.unwrap();
            prop_assert!(alignment.score >= 0);
            prop_assert!(alignment.alignment_statistics.identity >= 0.0);
            prop_assert!(alignment.alignment_statistics.identity <= 1.0);
        }
    }
}