use std::{
    sync::Arc,
    collections::{HashMap, BTreeMap},
    path::PathBuf,
};

use anyhow::{Result, Context};
use tokio::sync::{RwLock, Semaphore};
use futures::{Stream, StreamExt};
use serde::{Serialize, Deserialize};
use bio::alphabets::dna::revcomp;
use bio::pattern_matching::bom::BOM;
use bio::alignment::{pairwise, distance};

mod sequence;
mod analysis;
mod alignment;
mod variants;
mod cache;

pub use self::{
    sequence::{Sequence, SequenceMetadata, QualityScores},
    analysis::{AnalysisResult, AnalysisMetrics, SequenceStatistics},
    alignment::{AlignmentResult, AlignmentStrategy, AlignmentScores},
    variants::{Variant, VariantType, VariantAnnotation},
    cache::{Cache, CacheConfig, CacheMetrics},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomicsConfig {
    pub reference_genome: PathBuf,
    pub sequence_chunk_size: usize,
    pub max_concurrent_analyses: usize,
    pub quality_thresholds: QualityThresholds,
    pub alignment_params: AlignmentParameters,
    pub variant_calling: VariantCallingConfig,
    pub annotation_sources: Vec<AnnotationSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub min_base_quality: u8,
    pub min_mapping_quality: u8,
    pub min_depth: u32,
    pub max_error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentParameters {
    pub match_score: i32,
    pub mismatch_penalty: i32,
    pub gap_open_penalty: i32,
    pub gap_extend_penalty: i32,
    pub strategy: AlignmentStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantCallingConfig {
    pub min_variant_quality: f64,
    pub min_allele_frequency: f64,
    pub calling_algorithms: Vec<VariantCaller>,
    pub filters: Vec<VariantFilter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariantCaller {
    Bayesian { prior_probability: f64 },
    FrequencyBased { min_observations: u32 },
    MachineLearning { model_path: PathBuf },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantFilter {
    pub name: String,
    pub condition: FilterCondition,
    pub action: FilterAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterCondition {
    QualityBelow(f64),
    DepthBelow(u32),
    AlleleFrequencyBelow(f64),
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterAction {
    Exclude,
    Flag(String),
    ModifyQuality(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationSource {
    pub name: String,
    pub path: PathBuf,
    pub format: AnnotationFormat,
    pub priority: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AnnotationFormat {
    VEP,
    SnpEff,
    Custom,
}

pub struct GenomicsProcessor {
    config: GenomicsConfig,
    reference: Arc<ReferenceGenome>,
    cache: Arc<Cache>,
    sequence_processor: Arc<SequenceProcessor>,
    variant_caller: Arc<VariantCaller>,
    annotation_engine: Arc<AnnotationEngine>,
    alignment_semaphore: Arc<Semaphore>,
    metrics: Arc<RwLock<ProcessingMetrics>>,
}

impl GenomicsProcessor {
    pub async fn new(config: GenomicsConfig) -> Result<Self> {
        let reference = ReferenceGenome::load(&config.reference_genome).await?;
        let cache = Cache::new(CacheConfig::default())?;
        let sequence_processor = SequenceProcessor::new(&config);
        let variant_caller = Self::initialize_variant_caller(&config.variant_calling)?;
        let annotation_engine = AnnotationEngine::new(&config.annotation_sources).await?;
        
        Ok(Self {
            config,
            reference: Arc::new(reference),
            cache: Arc::new(cache),
            sequence_processor: Arc::new(sequence_processor),
            variant_caller: Arc::new(variant_caller),
            annotation_engine: Arc::new(annotation_engine),
            alignment_semaphore: Arc::new(Semaphore::new(config.max_concurrent_analyses)),
            metrics: Arc::new(RwLock::new(ProcessingMetrics::default())),
        })
    }

    pub async fn process_sequence_batch<S>(&self, sequences: S) -> Result<Vec<ProcessedSequence>>
    where
        S: Stream<Item = Result<Sequence>> + Send + 'static,
    {
        let mut results = Vec::new();
        let mut sequence_stream = sequences.chunks(self.config.sequence_chunk_size);

        while let Some(chunk) = sequence_stream.next().await {
            let _permit = self.alignment_semaphore.acquire().await?;
            let processed = self.process_chunk(chunk).await?;
            results.extend(processed);
        }

        Ok(results)
    }

    async fn process_chunk(&self, sequences: Vec<Result<Sequence>>) -> Result<Vec<ProcessedSequence>> {
        let sequences: Result<Vec<_>> = sequences.into_iter().collect();
        let sequences = sequences?;

        let mut processed = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            let alignment = self.align_sequence(&sequence).await?;
            let variants = self.call_variants(&alignment).await?;
            let annotated_variants = self.annotate_variants(variants).await?;
            
            processed.push(ProcessedSequence {
                sequence,
                alignment,
                variants: annotated_variants,
            });
        }

        Ok(processed)
    }

    async fn align_sequence(&self, sequence: &Sequence) -> Result<AlignmentResult> {
        let start = std::time::Instant::now();
        let cache_key = sequence.calculate_hash();

        if let Some(cached) = self.cache.get(&cache_key).await? {
            self.update_metrics(|m| m.cache_hits += 1).await;
            return Ok(cached);
        }

        let alignment = match self.config.alignment_params.strategy {
            AlignmentStrategy::Global => {
                pairwise::global_alignment(
                    sequence.bases(),
                    self.reference.sequence(),
                    self.config.alignment_params.match_score,
                    self.config.alignment_params.mismatch_penalty,
                    self.config.alignment_params.gap_open_penalty,
                )
            }
            AlignmentStrategy::Local => {
                pairwise::local_alignment(
                    sequence.bases(),
                    self.reference.sequence(),
                    self.config.alignment_params.match_score,
                    self.config.alignment_params.mismatch_penalty,
                    self.config.alignment_params.gap_open_penalty,
                )
            }
            AlignmentStrategy::SemiGlobal => {
                pairwise::semi_global_alignment(
                    sequence.bases(),
                    self.reference.sequence(),
                    self.config.alignment_params.match_score,
                    self.config.alignment_params.mismatch_penalty,
                    self.config.alignment_params.gap_open_penalty,
                )
            }
        };

        let result = AlignmentResult::from_alignment(alignment, sequence)?;
        self.cache.set(cache_key, &result).await?;
        
        self.update_metrics(|m| {
            m.alignments_performed += 1;
            m.total_alignment_time += start.elapsed();
        }).await;

        Ok(result)
    }

    async fn call_variants(&self, alignment: &AlignmentResult) -> Result<Vec<Variant>> {
        let variants = self.variant_caller.call_variants(alignment).await?;
        
        let filtered: Vec<_> = variants.into_iter()
            .filter(|v| self.apply_filters(v))
            .collect();

        self.update_metrics(|m| {
            m.variants_called += filtered.len();
        }).await;

        Ok(filtered)
    }

    async fn annotate_variants(&self, variants: Vec<Variant>) -> Result<Vec<AnnotatedVariant>> {
        self.annotation_engine.annotate_variants(variants).await
    }

    fn apply_filters(&self, variant: &Variant) -> bool {
        for filter in &self.config.variant_calling.filters {
            match &filter.condition {
                FilterCondition::QualityBelow(threshold) => {
                    if variant.quality < *threshold {
                        return false;
                    }
                }
                FilterCondition::DepthBelow(threshold) => {
                    if variant.depth < *threshold {
                        return false;
                    }
                }
                FilterCondition::AlleleFrequencyBelow(threshold) => {
                    if variant.allele_frequency < *threshold {
                        return false;
                    }
                }
                FilterCondition::Custom(expr) => {
                    if !self.evaluate_custom_filter(expr, variant)? {
                        return false;
                    }
                }
            }
        }
        true
    }

    async fn update_metrics<F>(&self, f: F)
    where
        F: FnOnce(&mut ProcessingMetrics),
    {
        let mut metrics = self.metrics.write().await;
        f(&mut metrics);
    }
}

#[derive(Debug, Default)]
struct ProcessingMetrics {
    sequences_processed: usize,
    alignments_performed: usize,
    variants_called: usize,
    cache_hits: usize,
    total_alignment_time: std::time::Duration,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessedSequence {
    pub sequence: Sequence,
    pub alignment: AlignmentResult,
    pub variants: Vec<AnnotatedVariant>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnnotatedVariant {
    pub variant: Variant,
    pub annotations: Vec<VariantAnnotation>,
    pub clinical_significance: Option<ClinicalSignificance>,
    pub population_frequency: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ClinicalSignificance {
    Pathogenic,
    LikelyPathogenic,
    Uncertain,
    LikelyBenign,
    Benign,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_sequence_processing() {
        let dir = tempdir().unwrap();
        let reference_path = dir.path().join("reference.fa");
        
        // Create test reference genome
        fs::write(&reference_path, b">ref\nACGTACGT\n").await.unwrap();
        
        let config = GenomicsConfig {
            reference_genome: reference_path,
            sequence_chunk_size: 1000,
            max_concurrent_analyses: 4,
            quality_thresholds: QualityThresholds {
                min_base_quality: 20,
                min_mapping_quality: 30,
                min_depth: 10,
                max_error_rate: 0.01,
            },
            alignment_params: AlignmentParameters {
                match_score: 2,
                mismatch_penalty: -1,
                gap_open_penalty: -2,
                gap_extend_penalty: -1,
                strategy: AlignmentStrategy::Global,
            },
            variant_calling: VariantCallingConfig {
                min_variant_quality: 30.0,
                min_allele_frequency: 0.05,
                calling_algorithms: vec![],
                filters: vec![],
            },
            annotation_sources: vec![],
        };

        let processor = GenomicsProcessor::new(config).await.unwrap();
        
        let sequences = stream::iter(vec![
            Ok(Sequence::new("test", b"ACGTACGT".to_vec(), None)),
            Ok(Sequence::new("test2", b"ACGTACGA".to_vec(), None)),
        ]);

        let results = processor.process_sequence_batch(sequences).await.unwrap();
        assert_eq!(results.len(), 2);
    }
}