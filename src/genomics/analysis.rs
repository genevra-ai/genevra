use std::{
    collections::{HashMap, BTreeMap},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Result, Context, bail};
use tokio::sync::{RwLock, Semaphore};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use bio::stats::{LogProb, Prob};
use ndarray::{Array1, Array2};
use statrs::distribution::{Normal, ContinuousCDF};
use futures::{stream::FuturesUnordered, StreamExt};

use crate::sequence::{Sequence, QualityScores};
use crate::alignment::AlignmentResult;
use super::variants::Variant;

const MIN_COVERAGE_DEPTH: u32 = 10;
const MAX_P_VALUE: f64 = 0.05;
const WINDOW_SIZE: usize = 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub sequence_stats: SequenceStatistics,
    pub coverage_analysis: CoverageAnalysis,
    pub quality_metrics: QualityMetrics,
    pub variant_analysis: VariantAnalysis,
    pub structural_features: StructuralFeatures,
    pub methylation_analysis: Option<MethylationAnalysis>,
    pub expression_data: Option<ExpressionData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceStatistics {
    pub length: usize,
    pub gc_content: f64,
    pub gc_skew: f64,
    pub complexity_score: f64,
    pub repeat_content: RepeatContent,
    pub kmer_frequencies: HashMap<Vec<u8>, f64>,
    pub codon_usage: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageAnalysis {
    pub mean_coverage: f64,
    pub median_coverage: f64,
    pub std_dev: f64,
    pub coverage_distribution: Vec<u32>,
    pub low_coverage_regions: Vec<Region>,
    pub high_coverage_regions: Vec<Region>,
    pub coverage_uniformity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub base_qualities: QualityDistribution,
    pub mapping_qualities: QualityDistribution,
    pub error_rates: ErrorRates,
    pub quality_scores: Vec<f64>,
    pub phred_scores: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantAnalysis {
    pub variant_density: f64,
    pub transition_transversion_ratio: f64,
    pub heterozygosity: f64,
    pub variant_type_distribution: HashMap<String, usize>,
    pub mutation_spectrum: MutationSpectrum,
    pub linkage_disequilibrium: Option<LinkageData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralFeatures {
    pub repeats: RepeatAnalysis,
    pub motifs: Vec<MotifOccurrence>,
    pub secondary_structure: Option<SecondaryStructure>,
    pub regulatory_elements: Vec<RegulatoryElement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethylationAnalysis {
    pub methylation_levels: Vec<f64>,
    pub methylated_sites: Vec<MethylationSite>,
    pub methylation_patterns: Vec<MethylationPattern>,
    pub differential_methylation: Option<DifferentialMethylation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionData {
    pub expression_levels: HashMap<String, f64>,
    pub differential_expression: Vec<DifferentialExpression>,
    pub correlation_matrix: Array2<f64>,
    pub cluster_assignments: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetrics {
    pub processing_time: Duration,
    pub memory_usage: usize,
    pub cpu_utilization: f64,
    pub error_rate: f64,
    pub confidence_scores: Vec<f64>,
}

pub struct SequenceAnalyzer {
    config: AnalysisConfig,
    metrics: Arc<RwLock<AnalysisMetrics>>,
    processing_semaphore: Arc<Semaphore>,
    cache: Arc<analysiscache::AnalysisCache>,
}

impl SequenceAnalyzer {
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(AnalysisMetrics::default())),
            processing_semaphore: Arc::new(Semaphore::new(num_cpus::get())),
            cache: Arc::new(analysiscache::AnalysisCache::new()),
        }
    }

    pub async fn analyze_sequence(&self, sequence: &Sequence) -> Result<AnalysisResult> {
        let start_time = Instant::now();
        let _permit = self.processing_semaphore.acquire().await?;

        if let Some(cached) = self.cache.get(sequence.calculate_hash()).await? {
            return Ok(cached);
        }

        let sequence_stats = self.compute_sequence_statistics(sequence)?;
        let coverage_analysis = self.analyze_coverage(sequence).await?;
        let quality_metrics = self.compute_quality_metrics(sequence)?;
        let variant_analysis = self.analyze_variants(sequence).await?;
        let structural_features = self.analyze_structural_features(sequence).await?;

        let methylation_analysis = if self.config.analyze_methylation {
            Some(self.analyze_methylation(sequence).await?)
        } else {
            None
        };

        let expression_data = if self.config.analyze_expression {
            Some(self.analyze_expression(sequence).await?)
        } else {
            None
        };

        let result = AnalysisResult {
            sequence_stats,
            coverage_analysis,
            quality_metrics,
            variant_analysis,
            structural_features,
            methylation_analysis,
            expression_data,
        };

        self.update_metrics(start_time.elapsed()).await?;
        self.cache.set(sequence.calculate_hash(), &result).await?;

        Ok(result)
    }

    async fn analyze_coverage(&self, sequence: &Sequence) -> Result<CoverageAnalysis> {
        let coverage_data = self.calculate_coverage_data(sequence)?;
        
        let mean_coverage = coverage_data.mean();
        let median_coverage = coverage_data.median();
        let std_dev = coverage_data.std_dev();

        let low_coverage_regions = self.identify_low_coverage_regions(&coverage_data)?;
        let high_coverage_regions = self.identify_high_coverage_regions(&coverage_data)?;
        let coverage_uniformity = self.calculate_coverage_uniformity(&coverage_data);

        Ok(CoverageAnalysis {
            mean_coverage,
            median_coverage,
            std_dev,
            coverage_distribution: coverage_data.into_raw_vec(),
            low_coverage_regions,
            high_coverage_regions,
            coverage_uniformity,
        })
    }

    fn compute_sequence_statistics(&self, sequence: &Sequence) -> Result<SequenceStatistics> {
        let bases = sequence.bases();
        let length = bases.len();

        let gc_content = bases.par_iter()
            .filter(|&&b| b == b'G' || b == b'C')
            .count() as f64 / length as f64;

        let gc_skew = self.calculate_gc_skew(bases);
        let complexity_score = self.calculate_complexity(bases);
        let repeat_content = self.analyze_repeats(bases)?;
        let kmer_frequencies = self.calculate_kmer_frequencies(bases, self.config.kmer_size);
        let codon_usage = self.analyze_codon_usage(bases);

        Ok(SequenceStatistics {
            length,
            gc_content,
            gc_skew,
            complexity_score,
            repeat_content,
            kmer_frequencies,
            codon_usage,
        })
    }

    async fn analyze_variants(&self, sequence: &Sequence) -> Result<VariantAnalysis> {
        let variants = self.call_variants(sequence).await?;
        
        let variant_density = variants.len() as f64 / sequence.len() as f64;
        let (transitions, transversions) = self.count_mutation_types(&variants);
        let tt_ratio = transitions as f64 / transversions as f64;
        
        let heterozygosity = self.calculate_heterozygosity(&variants);
        let variant_type_distribution = self.categorize_variants(&variants);
        let mutation_spectrum = self.analyze_mutation_spectrum(&variants);
        let linkage_disequilibrium = self.calculate_linkage_disequilibrium(&variants).await?;

        Ok(VariantAnalysis {
            variant_density,
            transition_transversion_ratio: tt_ratio,
            heterozygosity,
            variant_type_distribution,
            mutation_spectrum,
            linkage_disequilibrium,
        })
    }

    async fn analyze_structural_features(&self, sequence: &Sequence) -> Result<StructuralFeatures> {
        let repeats = self.analyze_repeat_structures(sequence).await?;
        let motifs = self.find_motifs(sequence).await?;
        let secondary_structure = if self.config.predict_secondary_structure {
            Some(self.predict_secondary_structure(sequence).await?)
        } else {
            None
        };
        let regulatory_elements = self.identify_regulatory_elements(sequence).await?;

        Ok(StructuralFeatures {
            repeats,
            motifs,
            secondary_structure,
            regulatory_elements,
        })
    }

    async fn analyze_methylation(&self, sequence: &Sequence) -> Result<MethylationAnalysis> {
        let methylation_sites = self.identify_methylation_sites(sequence).await?;
        let methylation_levels = self.calculate_methylation_levels(&methylation_sites);
        let methylation_patterns = self.analyze_methylation_patterns(&methylation_sites);
        let differential_methylation = self.analyze_differential_methylation(&methylation_sites).await?;

        Ok(MethylationAnalysis {
            methylation_levels,
            methylated_sites: methylation_sites,
            methylation_patterns,
            differential_methylation,
        })
    }

    async fn analyze_expression(&self, sequence: &Sequence) -> Result<ExpressionData> {
        let expression_levels = self.calculate_expression_levels(sequence).await?;
        let differential_expression = self.analyze_differential_expression(&expression_levels).await?;
        let correlation_matrix = self.calculate_expression_correlation(&expression_levels);
        let cluster_assignments = self.cluster_expression_patterns(&expression_levels);

        Ok(ExpressionData {
            expression_levels,
            differential_expression,
            correlation_matrix,
            cluster_assignments,
        })
    }

    async fn update_metrics(&self, processing_time: Duration) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        metrics.processing_time += processing_time;
        metrics.cpu_utilization = self.measure_cpu_utilization().await?;
        metrics.memory_usage = self.measure_memory_usage()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_sequence_analysis() {
        let sequence = Sequence::new("test", b"ACGTACGT".to_vec(), None);
        let config = AnalysisConfig::default();
        let analyzer = SequenceAnalyzer::new(config);
        
        let result = analyzer.analyze_sequence(&sequence).await.unwrap();
        assert!(result.sequence_stats.gc_content > 0.0);
        assert!(result.coverage_analysis.mean_coverage > 0.0);
    }

    #[tokio::test]
    async fn test_variant_analysis() {
        let sequence = Sequence::new("test", b"ACGTACGT".to_vec(), None);
        let config = AnalysisConfig::default();
        let analyzer = SequenceAnalyzer::new(config);
        
        let variants = analyzer.analyze_variants(&sequence).await.unwrap();
        assert!(variants.variant_density >= 0.0);
        assert!(variants.transition_transversion_ratio > 0.0);
    }
}