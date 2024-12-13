#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tokio::sync::{RwLock, Semaphore, broadcast};
use anyhow::{Result, Context};
use futures::stream::{self, StreamExt};
use rayon::prelude::*;

pub mod ai;
pub mod genomics;
pub mod utils;
pub mod cli;
pub mod config;
pub mod error;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const MAX_PENDING_OPERATIONS: usize = 10_000;
static INITIALIZED: AtomicBool = AtomicBool::new(false);

#[derive(Debug)]
pub struct GlobalState {
    config: Config,
    genome_cache: genomics::Cache,
    model_registry: ai::ModelRegistry,
    processing_queue: Arc<RwLock<utils::Queue>>,
    operation_semaphore: Arc<Semaphore>,
    shutdown_signal: broadcast::Sender<()>,
    _gpu_guard: Option<Arc<cuda::GpuGuard>>,
}

impl GlobalState {
    pub async fn new(config: Config) -> Result<Self> {
        let genome_cache = genomics::Cache::with_capacity(config.cache_size)
            .await
            .context("failed to initialize genome cache")?;
            
        let model_registry = ai::ModelRegistry::new(&config.model_path)
            .await
            .context("failed to initialize model registry")?;
            
        let processing_queue = Arc::new(RwLock::new(utils::Queue::with_capacity(config.queue_size)));
        let operation_semaphore = Arc::new(Semaphore::new(MAX_PENDING_OPERATIONS));
        let (shutdown_signal, _) = broadcast::channel(1);
        
        let gpu_guard = if config.use_gpu {
            Some(Arc::new(cuda::GpuGuard::new()?))
        } else {
            None
        };

        Ok(Self {
            config,
            genome_cache,
            model_registry,
            processing_queue,
            operation_semaphore,
            shutdown_signal,
            _gpu_guard: gpu_guard,
        })
    }

    pub fn acquire_operation_permit(&self) -> Result<tokio::sync::SemaphorePermit<'_>> {
        Ok(self.operation_semaphore.try_acquire()?)
    }

    pub async fn process_batch<T>(&self, items: Vec<T>) -> Result<Vec<ProcessedResult>>
    where
        T: Send + Sync + 'static,
        ProcessedResult: From<T>,
    {
        let permits = stream::iter(0..items.len())
            .map(|_| self.operation_semaphore.clone().acquire_owned())
            .buffer_unordered(self.config.max_threads)
            .collect::<Vec<_>>()
            .await;

        let results = items.into_par_iter()
            .zip(permits)
            .map(|(item, permit)| {
                let result = ProcessedResult::from(item);
                drop(permit);
                result
            })
            .collect();

        Ok(results)
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Config {
    pub max_threads: usize,
    pub model_path: std::path::PathBuf,
    pub use_gpu: bool,
    pub cache_size: usize,
    pub queue_size: usize,
    pub genomics: GenomicsConfig,
    pub ai: AIConfig,
    pub advanced: AdvancedConfig,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct GenomicsConfig {
    pub min_quality_score: f64,
    pub min_sequence_length: usize,
    pub max_sequence_length: usize,
    pub reference_genome: std::path::PathBuf,
    pub alignment_algorithm: AlignmentAlgorithm,
    pub kmer_size: usize,
    pub max_edit_distance: usize,
    pub sequence_compression: CompressionType,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct AIConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub architecture: ModelArchitecture,
    pub optimizer: OptimizerConfig,
    pub loss_function: LossFunction,
    pub gradient_accumulation_steps: usize,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct AdvancedConfig {
    pub memory_limit: usize,
    pub io_threads: usize,
    pub prefetch_factor: usize,
    pub worker_threads: usize,
    pub watchdog_interval_ms: u64,
    pub fallback_strategy: FallbackStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum AlignmentAlgorithm {
    SmithWaterman { gap_penalty: i32, match_score: i32 },
    NeedlemanWunsch { gap_open: i32, gap_extend: i32 },
    Blast { word_size: usize, threshold: f64 },
    Custom(u32),
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ModelArchitecture {
    Transformer {
        num_heads: usize,
        num_layers: usize,
        hidden_dim: usize,
        dropout: f32,
        activation: ActivationType,
    },
    Convolutional {
        num_layers: usize,
        filters: Vec<usize>,
        kernel_sizes: Vec<usize>,
        stride: Vec<usize>,
        dilation: Vec<usize>,
    },
    Hybrid {
        conv_layers: Vec<ConvSpec>,
        transformer_layers: Vec<TransformerSpec>,
        aggregation_type: AggregationType,
    },
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ProcessedResult {
    pub id: uuid::Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metrics: std::collections::HashMap<String, f64>,
    pub status: ProcessingStatus,
    pub artifacts: Vec<Arc<ProcessingArtifact>>,
}

impl<T> From<T> for ProcessedResult
where
    T: Send + Sync + 'static,
{
    fn from(_item: T) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            metrics: std::collections::HashMap::new(),
            status: ProcessingStatus::Completed,
            artifacts: Vec::new(),
        }
    }
}

pub async fn init_with_config(config: Config) -> Result<Arc<GlobalState>> {
    if INITIALIZED.swap(true, Ordering::SeqCst) {
        anyhow::bail!("library already initialized");
    }

    tracing_subscriber::fmt::try_init().ok();

    if config.use_gpu {
        cuda::init_cuda()?;
    }

    let state = GlobalState::new(config).await?;
    let state = Arc::new(state);

    genomics::init(Arc::clone(&state))?;
    ai::init(Arc::clone(&state)).await?;
    utils::init(Arc::clone(&state))?;

    Ok(state)
}

pub async fn init() -> Result<Arc<GlobalState>> {
    init_with_config(Config::default()).await
}

pub async fn shutdown(state: Arc<GlobalState>) -> Result<()> {
    let _ = state.shutdown_signal.send(());
    
    let queue = state.processing_queue().read().await;
    queue.wait_empty().await?;
    
    utils::shutdown(Arc::clone(&state))?;
    ai::shutdown(Arc::clone(&state)).await?;
    genomics::shutdown(Arc::clone(&state))?;

    INITIALIZED.store(false, Ordering::SeqCst);
    Ok(())
}

mod cuda {
    use super::*;
    
    pub struct GpuGuard(());
    
    impl GpuGuard {
        pub fn new() -> Result<Self> {
            init_cuda()?;
            Ok(Self(()))
        }
    }
    
    pub fn init_cuda() -> Result<()> {
        if tch::Cuda::is_available() {
            unsafe { tch::utils::init_cuda() };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[tokio::test]
    async fn test_init_and_shutdown() {
        let state = init().await.unwrap();
        shutdown(state).await.unwrap();
    }

    #[test_case(AlignmentAlgorithm::SmithWaterman { gap_penalty: -1, match_score: 2 } => true)]
    #[test_case(AlignmentAlgorithm::NeedlemanWunsch { gap_open: -2, gap_extend: -1 } => true)]
    #[test_case(AlignmentAlgorithm::Blast { word_size: 11, threshold: 0.001 } => true)]
    fn test_alignment_algorithm_serde(algo: AlignmentAlgorithm) -> bool {
        let serialized = serde_json::to_string(&algo).unwrap();
        let deserialized: AlignmentAlgorithm = serde_json::from_str(&serialized).unwrap();
        algo == deserialized
    }

    proptest::proptest! {
        #[test]
        fn test_processed_result_creation(s in "\\PC*") {
            let result = ProcessedResult::from(s);
            assert!(result.metrics.is_empty());
        }
    }
}