use std::{
    collections::{HashMap, BTreeMap, HashSet},
    sync::Arc,
    path::PathBuf,
};

use anyhow::{Result, Context, bail};
use bio::stats::{LogProb, Prob};
use itertools::Itertools;
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

const MIN_VARIANT_QUALITY: f64 = 30.0;
const MIN_ALLELE_FREQUENCY: f64 = 0.05;
const MAX_STRAND_BIAS: f64 = 0.8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    pub position: usize,
    pub reference_allele: Vec<u8>,
    pub alternate_alleles: Vec<Vec<u8>>,
    pub quality: f64,
    pub filter_status: FilterStatus,
    pub genotype: Option<Genotype>,
    pub variant_type: VariantType,
    pub annotations: Vec<VariantAnnotation>,
    pub metrics: VariantMetrics,
    pub statistics: VariantStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantAnnotation {
    pub source: String,
    pub impact: Impact,
    pub consequence: Vec<String>,
    pub gene_id: Option<String>,
    pub transcript_id: Option<String>,
    pub protein_change: Option<String>,
    pub clinical_significance: Option<ClinicalSignificance>,
    pub population_frequencies: HashMap<String, f64>,
    pub conservation_scores: ConservationScores,
    pub functional_predictions: FunctionalPredictions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantMetrics {
    pub depth: u32,
    pub allele_frequency: f64,
    pub strand_bias: f64,
    pub mapping_quality: f64,
    pub base_qualities: Vec<u8>,
    pub read_position_bias: f64,
    pub haplotype_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantStatistics {
    pub transition_transversion: bool,
    pub zygosity: Zygosity,
    pub hardy_weinberg: Option<HardyWeinbergStats>,
    pub linkage_disequilibrium: Option<LinkageStats>,
    pub mutation_spectrum: MutationSpectrum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VariantType {
    SNV,
    Insertion,
    Deletion,
    MNV,
    Complex,
    StructuralVariant(SVType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SVType {
    Deletion,
    Duplication,
    Inversion,
    Translocation,
    InsertionMobile,
    ComplexRearrangement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Impact {
    High,
    Moderate,
    Low,
    Modifier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterStatus {
    Pass,
    Fail(FilterReason),
    Warning(FilterReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterReason {
    LowQuality,
    StrandBias,
    LowDepth,
    LowMappingQuality,
    RepeatRegion,
    PositionBias,
    Custom(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genotype {
    pub alleles: Vec<usize>,
    pub phased: bool,
    pub likelihood: Vec<f64>,
    pub posterior_probability: f64,
    pub genotype_quality: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Zygosity {
    Homozygous,
    Heterozygous,
    HomozygousReference,
    HomozygousAlternate,
    Hemizygous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationScores {
    pub phylop: Option<f64>,
    pub phastcons: Option<f64>,
    pub gerp: Option<f64>,
    pub sift: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionalPredictions {
    pub polyphen: Option<PredictionScore>,
    pub sift: Option<PredictionScore>,
    pub cadd: Option<f64>,
    pub dann: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionScore {
    pub score: f64,
    pub prediction: PredictionClass,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionClass {
    Benign,
    PossiblyDamaging,
    ProbablyDamaging,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardyWeinbergStats {
    pub observed_homozygous_ref: u32,
    pub observed_heterozygous: u32,
    pub observed_homozygous_alt: u32,
    pub expected_homozygous_ref: f64,
    pub expected_heterozygous: f64,
    pub expected_homozygous_alt: f64,
    pub chi_square: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkageStats {
    pub r_squared: f64,
    pub d_prime: f64,
    pub lod_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationSpectrum {
    pub transition_counts: HashMap<(char, char), u32>,
    pub transversion_counts: HashMap<(char, char), u32>,
    pub context_frequencies: HashMap<String, f64>,
}

pub struct VariantCaller {
    config: VariantCallerConfig,
    reference: Arc<ReferenceGenome>,
    annotation_db: Arc<AnnotationDatabase>,
    metrics: Arc<RwLock<CallerMetrics>>,
}

impl VariantCaller {
    pub fn new(
        config: VariantCallerConfig,
        reference: Arc<ReferenceGenome>,
        annotation_db: Arc<AnnotationDatabase>,
    ) -> Self {
        Self {
            config,
            reference,
            annotation_db,
            metrics: Arc::new(RwLock::new(CallerMetrics::default())),
        }
    }

    pub async fn call_variants(&self, alignment: &AlignmentResult) -> Result<Vec<Variant>> {
        let start_time = std::time::Instant::now();
        let mut variants = Vec::new();

        // SNV and small indel calling
        let small_variants = self.call_small_variants(alignment)?;
        variants.extend(small_variants);

        // Structural variant calling if enabled
        if self.config.enable_sv_calling {
            let structural_variants = self.call_structural_variants(alignment).await?;
            variants.extend(structural_variants);
        }

        // Filter and annotate variants
        let filtered_variants = self.filter_variants(variants).await?;
        let annotated_variants = self.annotate_variants(filtered_variants).await?;

        // Update metrics
        self.update_metrics(start_time.elapsed(), &annotated_variants).await?;

        Ok(annotated_variants)
    }

    fn call_small_variants(&self, alignment: &AlignmentResult) -> Result<Vec<Variant>> {
        let mut variants = Vec::new();
        let reference_sequence = self.reference.sequence();
        
        for position in 0..alignment.aligned_sequences.reference.len() {
            if let Some(variant) = self.detect_variant_at_position(
                position,
                alignment,
                reference_sequence,
            )? {
                variants.push(variant);
            }
        }

        Ok(variants)
    }

    async fn call_structural_variants(&self, alignment: &AlignmentResult) -> Result<Vec<Variant>> {
        let mut sv_variants = Vec::new();
        
        // Split-read analysis
        let split_read_variants = self.detect_split_read_variants(alignment)?;
        sv_variants.extend(split_read_variants);

        // Read-pair analysis
        if self.config.enable_read_pair_analysis {
            let read_pair_variants = self.detect_read_pair_variants(alignment)?;
            sv_variants.extend(read_pair_variants);
        }

        // Coverage analysis for CNVs
        if self.config.enable_cnv_detection {
            let cnv_variants = self.detect_copy_number_variants(alignment).await?;
            sv_variants.extend(cnv_variants);
        }

        Ok(sv_variants)
    }

    async fn filter_variants(&self, variants: Vec<Variant>) -> Result<Vec<Variant>> {
        let filtered: Vec<_> = variants.into_par_iter()
            .filter(|variant| self.apply_filters(variant))
            .collect();

        Ok(filtered)
    }

    async fn annotate_variants(&self, variants: Vec<Variant>) -> Result<Vec<Variant>> {
        let mut annotated_variants = Vec::with_capacity(variants.len());

        for variant in variants {
            let mut annotated = variant.clone();
            
            // Functional annotation
            let functional_annotations = self.annotation_db.get_functional_annotations(&variant).await?;
            annotated.annotations.extend(functional_annotations);

            // Population frequencies
            if let Some(frequencies) = self.annotation_db.get_population_frequencies(&variant).await? {
                for annotation in &mut annotated.annotations {
                    annotation.population_frequencies.extend(frequencies.clone());
                }
            }

            // Clinical significance
            if let Some(clinical_data) = self.annotation_db.get_clinical_data(&variant).await? {
                for annotation in &mut annotated.annotations {
                    annotation.clinical_significance = Some(clinical_data.clone());
                }
            }

            annotated_variants.push(annotated);
        }

        Ok(annotated_variants)
    }

    fn apply_filters(&self, variant: &Variant) -> bool {
        // Basic quality filters
        if variant.quality < MIN_VARIANT_QUALITY {
            return false;
        }

        if variant.metrics.allele_frequency < MIN_ALLELE_FREQUENCY {
            return false;
        }

        if variant.metrics.strand_bias > MAX_STRAND_BIAS {
            return false;
        }

        // Custom filters from configuration
        for filter in &self.config.custom_filters {
            if !filter.apply(variant) {
                return false;
            }
        }

        true
    }

    fn detect_variant_at_position(
        &self,
        position: usize,
        alignment: &AlignmentResult,
        reference_sequence: &[u8],
    ) -> Result<Option<Variant>> {
        let reference_base = reference_sequence[position];
        let aligned_base = alignment.aligned_sequences.query[position];

        if reference_base != aligned_base {
            let variant = self.create_variant(
                position,
                reference_base,
                aligned_base,
                alignment,
            )?;

            Ok(Some(variant))
        } else {
            Ok(None)
        }
    }

    fn create_variant(
        &self,
        position: usize,
        reference_base: u8,
        alternate_base: u8,
        alignment: &AlignmentResult,
    ) -> Result<Variant> {
        let variant_type = self.determine_variant_type(reference_base, alternate_base);
        let metrics = self.calculate_variant_metrics(position, alignment)?;
        let statistics = self.calculate_variant_statistics(position, alignment)?;

        Ok(Variant {
            position,
            reference_allele: vec![reference_base],
            alternate_alleles: vec![vec![alternate_base]],
            quality: self.calculate_variant_quality(position, alignment)?,
            filter_status: FilterStatus::Pass,
            genotype: self.determine_genotype(position, alignment)?,
            variant_type,
            annotations: Vec::new(),
            metrics,
            statistics,
        })
    }

    async fn update_metrics(&self, elapsed: std::time::Duration, variants: &[Variant]) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        metrics.variants_processed += variants.len();
        metrics.processing_time += elapsed;
        metrics.total_variants += variants.len();
        
        for variant in variants {
            match variant.variant_type {
                VariantType::SNV => metrics.snv_count += 1,
                VariantType::Insertion => metrics.insertion_count += 1,
                VariantType::Deletion => metrics.deletion_count += 1,
                VariantType::MNV => metrics.mnv_count += 1,
                VariantType::Complex => metrics.complex_count += 1,
                VariantType::StructuralVariant(_) => metrics.sv_count += 1,
            }
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
struct CallerMetrics {
    variants_processed: usize,
    processing_time: std::time::Duration,
    total_variants: usize,
    snv_count: usize,
    insertion_count: usize,
    deletion_count: usize,
    mnv_count: usize,
    complex_count: usize,
    sv_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn test_variant_filtering() {
        let config = VariantCallerConfig::default();
        let reference = Arc::new(ReferenceGenome::test_genome());
        let annotation_db = Arc::new(AnnotationDatabase::test_db());
        
        let caller = VariantCaller::new(config, reference, annotation_db);
        
        let variants = vec![
            Variant {
                quality: MIN_VARIANT_QUALITY - 1.0,
                ..Variant::test_variant()
            },
            Variant {
                quality: MIN_VARIANT_QUALITY + 10.0,
                ..Variant::test_variant()
            },
        ];

        let filtered = caller.filter_variants(variants).await.unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].quality > MIN_VARIANT_QUALITY);
    }

    #[tokio::test]
    async fn test_variant_annotation() {
        let config = VariantCallerConfig::default();
        let reference = Arc::new(ReferenceGenome::test_genome());
        let annotation_db = Arc::new(AnnotationDatabase::test_db());
        
        let caller = VariantCaller::new(config, reference, annotation_db);
        
        let variant = Variant::test_variant();
        let variants = vec![variant];
        
        let annotated = caller.annotate_variants(variants).await.unwrap();
        assert!(!annotated[0].annotations.is_empty());
        
        // Check functional annotations
        let functional_anno = &annotated[0].annotations[0];
        assert!(functional_anno.impact != Impact::Modifier);
        
        // Check population frequencies
        assert!(!functional_anno.population_frequencies.is_empty());
        
        // Check clinical significance
        assert!(functional_anno.clinical_significance.is_some());
    }

    #[tokio::test]
    async fn test_structural_variant_calling() {
        let mut config = VariantCallerConfig::default();
        config.enable_sv_calling = true;
        config.enable_read_pair_analysis = true;
        config.enable_cnv_detection = true;

        let reference = Arc::new(ReferenceGenome::test_genome());
        let annotation_db = Arc::new(AnnotationDatabase::test_db());
        
        let caller = VariantCaller::new(config, reference, annotation_db);
        let alignment = AlignmentResult::test_alignment_with_sv();
        
        let variants = caller.call_structural_variants(&alignment).await.unwrap();
        assert!(!variants.is_empty());
        
        // Check for different SV types
        let sv_types: HashSet<_> = variants.iter()
            .filter_map(|v| match v.variant_type {
                VariantType::StructuralVariant(sv_type) => Some(sv_type),
                _ => None,
            })
            .collect();
        
        assert!(sv_types.contains(&SVType::Deletion));
        assert!(sv_types.contains(&SVType::Duplication));
    }

    #[test]
    fn test_variant_metrics() {
        let variant = Variant::test_variant();
        
        assert!(variant.metrics.depth > 0);
        assert!(variant.metrics.allele_frequency > 0.0);
        assert!(variant.metrics.allele_frequency <= 1.0);
        assert!(variant.metrics.mapping_quality > 0.0);
    }

    #[test]
    fn test_variant_statistics() {
        let variant = Variant::test_variant();
        
        // Check Hardy-Weinberg equilibrium stats
        if let Some(hw_stats) = &variant.statistics.hardy_weinberg {
            assert!(hw_stats.p_value >= 0.0);
            assert!(hw_stats.p_value <= 1.0);
            assert!(hw_stats.chi_square >= 0.0);
        }

        // Check linkage statistics
        if let Some(ld_stats) = &variant.statistics.linkage_disequilibrium {
            assert!(ld_stats.r_squared >= 0.0);
            assert!(ld_stats.r_squared <= 1.0);
            assert!(ld_stats.d_prime >= -1.0);
            assert!(ld_stats.d_prime <= 1.0);
        }
    }

    #[test]
    fn test_prediction_scores() {
        let predictions = FunctionalPredictions {
            polyphen: Some(PredictionScore {
                score: 0.95,
                prediction: PredictionClass::ProbablyDamaging,
                confidence: Some(0.9),
            }),
            sift: Some(PredictionScore {
                score: 0.02,
                prediction: PredictionClass::Benign,
                confidence: Some(0.85),
            }),
            cadd: Some(25.5),
            dann: Some(0.98),
        };

        assert!(predictions.polyphen.as_ref().unwrap().score > 0.9);
        assert_eq!(
            predictions.polyphen.as_ref().unwrap().prediction,
            PredictionClass::ProbablyDamaging
        );
        assert!(predictions.cadd.unwrap() > 20.0);
    }

    proptest::proptest! {
        #[test]
        fn test_variant_creation_properties(
            position in 0usize..1000usize,
            quality in 0.0f64..100.0f64,
            depth in 1u32..1000u32,
            allele_freq in 0.0f64..1.0f64
        ) {
            let variant = Variant {
                position,
                quality,
                metrics: VariantMetrics {
                    depth,
                    allele_frequency: allele_freq,
                    ..VariantMetrics::default()
                },
                ..Variant::test_variant()
            };

            prop_assert!(variant.position == position);
            prop_assert!(variant.quality == quality);
            prop_assert!(variant.metrics.depth == depth);
            prop_assert!((variant.metrics.allele_frequency - allele_freq).abs() < f64::EPSILON);
        }
    }

    impl Variant {
        fn test_variant() -> Self {
            Self {
                position: 100,
                reference_allele: vec![b'A'],
                alternate_alleles: vec![vec![b'T']],
                quality: 60.0,
                filter_status: FilterStatus::Pass,
                genotype: Some(Genotype {
                    alleles: vec![0, 1],
                    phased: false,
                    likelihood: vec![-10.0, -5.0, -10.0],
                    posterior_probability: 0.95,
                    genotype_quality: 30,
                }),
                variant_type: VariantType::SNV,
                annotations: Vec::new(),
                metrics: VariantMetrics {
                    depth: 30,
                    allele_frequency: 0.5,
                    strand_bias: 0.1,
                    mapping_quality: 60.0,
                    base_qualities: vec![30, 30, 30],
                    read_position_bias: 0.1,
                    haplotype_score: 0.8,
                },
                statistics: VariantStatistics {
                    transition_transversion: true,
                    zygosity: Zygosity::Heterozygous,
                    hardy_weinberg: Some(HardyWeinbergStats {
                        observed_homozygous_ref: 25,
                        observed_heterozygous: 50,
                        observed_homozygous_alt: 25,
                        expected_homozygous_ref: 25.0,
                        expected_heterozygous: 50.0,
                        expected_homozygous_alt: 25.0,
                        chi_square: 0.0,
                        p_value: 1.0,
                    }),
                    linkage_disequilibrium: Some(LinkageStats {
                        r_squared: 0.8,
                        d_prime: 0.9,
                        lod_score: 5.0,
                    }),
                    mutation_spectrum: MutationSpectrum {
                        transition_counts: HashMap::new(),
                        transversion_counts: HashMap::new(),
                        context_frequencies: HashMap::new(),
                    },
                },
            }
        }
    }

    impl VariantMetrics {
        fn default() -> Self {
            Self {
                depth: 30,
                allele_frequency: 0.5,
                strand_bias: 0.1,
                mapping_quality: 60.0,
                base_qualities: vec![30],
                read_position_bias: 0.1,
                haplotype_score: 0.8,
            }
        }
    }
}