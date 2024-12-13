use std::{
    sync::Arc,
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};

use anyhow::{Result, Context, bail};
use futures::{Stream, StreamExt};
use tokio::sync::{mpsc, Mutex, RwLock};
use tch::{Tensor, Device, nn::{self, OptimizerConfig}};
use metrics::{counter, gauge, histogram};
use rayon::prelude::*;

use crate::models::{Model, ModelConfig, OptimizerType};
use super::metrics::MetricsCollector;

const GRADIENT_ACCUMULATION_BATCH_SIZE: usize = 32;
const DEFAULT_TRAINING_STEPS: usize = 10000;

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub max_steps: usize,
    pub gradient_accumulation_steps: usize,
    pub max_gradient_norm: f64,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub evaluation_steps: usize,
    pub checkpoint_steps: usize,
    pub mixed_precision: bool,
    pub distributed: Option<DistributedConfig>,
    pub early_stopping: Option<EarlyStoppingConfig>,
}

#[derive(Debug, Clone)]
pub struct DistributedConfig {
    pub world_size: usize,
    pub rank: usize,
    pub backend: DistributedBackend,
    pub master_addr: String,
    pub master_port: u16,
}

#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f64,
    pub monitor_metric: String,
    pub mode: MonitorMode,
}

#[derive(Debug, Clone, Copy)]
pub enum MonitorMode {
    Min,
    Max,
}

#[derive(Debug)]
pub struct TrainingPipeline {
    config: TrainingConfig,
    optimizer: Arc<Mutex<Optimizer>>,
    scheduler: Option<Arc<Mutex<LRScheduler>>>,
    metrics: Arc<MetricsCollector>,
    device_manager: Arc<DeviceManager>,
    gradient_scaler: Option<Arc<Mutex<GradientScaler>>>,
}

#[derive(Debug)]
struct Optimizer {
    inner: Box<dyn OptimizerConfig>,
    params: Vec<nn::Parameter>,
    current_lr: f64,
}

#[derive(Debug)]
struct LRScheduler {
    scheduler_type: SchedulerType,
    initial_lr: f64,
    current_step: usize,
    total_steps: usize,
    warmup_steps: usize,
}

#[derive(Debug)]
struct GradientScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    consecutive_successes: usize,
}

impl TrainingPipeline {
    pub async fn new(
        config: TrainingConfig,
        device_manager: Arc<DeviceManager>,
        metrics: Arc<MetricsCollector>,
    ) -> Result<Self> {
        let optimizer = Arc::new(Mutex::new(Optimizer::new(
            config.learning_rate,
            config.max_gradient_norm,
        )));

        let scheduler = if config.warmup_steps > 0 {
            Some(Arc::new(Mutex::new(LRScheduler::new(
                config.learning_rate,
                config.max_steps,
                config.warmup_steps,
            ))))
        } else {
            None
        };

        let gradient_scaler = if config.mixed_precision {
            Some(Arc::new(Mutex::new(GradientScaler::new())))
        } else {
            None
        };

        Ok(Self {
            config,
            optimizer,
            scheduler,
            metrics,
            device_manager,
            gradient_scaler,
        })
    }

    pub async fn train<D>(&self, model: Arc<dyn Model>, mut dataset: D) -> Result<TrainingMetrics>
    where
        D: Stream<Item = Result<Tensor>> + Send + 'static,
    {
        let start_time = Instant::now();
        let mut training_metrics = TrainingMetrics::new();
        let mut early_stopper = self.config.early_stopping
            .as_ref()
            .map(EarlyStopper::new);

        let (tx, mut rx) = mpsc::channel(self.config.batch_size);
        let data_loader = tokio::spawn(async move {
            while let Some(batch) = dataset.next().await {
                if tx.send(batch).await.is_err() {
                    break;
                }
            }
        });

        let mut accumulated_gradients = Vec::new();
        let mut step = 0;

        while let Some(batch_result) = rx.recv().await {
            let batch = batch_result?;
            
            // Mixed precision training
            let loss = if self.config.mixed_precision {
                self.train_step_mixed_precision(&model, &batch).await?
            } else {
                self.train_step(&model, &batch).await?
            };

            accumulated_gradients.push(loss.backward());

            // Gradient accumulation
            if accumulated_gradients.len() >= self.config.gradient_accumulation_steps {
                self.apply_accumulated_gradients(&accumulated_gradients).await?;
                accumulated_gradients.clear();

                // Update learning rate
                if let Some(scheduler) = &self.scheduler {
                    scheduler.lock().await.step();
                }

                step += 1;
                training_metrics.update(loss.double_value(&[]), step);

                // Evaluation
                if step % self.config.evaluation_steps == 0 {
                    let eval_metrics = self.evaluate(&model).await?;
                    training_metrics.add_evaluation(eval_metrics);

                    // Early stopping check
                    if let Some(stopper) = &mut early_stopper {
                        if stopper.should_stop(&eval_metrics) {
                            break;
                        }
                    }
                }

                // Checkpointing
                if step % self.config.checkpoint_steps == 0 {
                    self.save_checkpoint(&model, &training_metrics, step).await?;
                }
            }

            // Record metrics
            histogram!("training_loss", loss.double_value(&[]));
            gauge!("learning_rate", self.optimizer.lock().await.current_lr);
            counter!("training_steps", 1);

            if step >= self.config.max_steps {
                break;
            }
        }

        data_loader.abort();
        training_metrics.training_time = start_time.elapsed();

        Ok(training_metrics)
    }

    async fn train_step(&self, model: &Arc<dyn Model>, batch: &Tensor) -> Result<Tensor> {
        let outputs = model.forward(batch).await?;
        let loss = self.compute_loss(&outputs, batch);

        if !loss.requires_grad() {
            bail!("Loss does not require gradients");
        }

        Ok(loss)
    }

    async fn train_step_mixed_precision(
        &self,
        model: &Arc<dyn Model>,
        batch: &Tensor,
    ) -> Result<Tensor> {
        let mut scaler = self.gradient_scaler.as_ref().unwrap().lock().await;
        
        // Forward pass in half precision
        let outputs = {
            let _half = tch::autocast(true);
            model.forward(batch).await?
        };

        // Compute loss in full precision
        let loss = self.compute_loss(&outputs, batch);
        
        // Scale loss and backward pass
        let scaled_loss = loss * scaler.scale;
        scaled_loss.backward();

        // Unscale gradients and check for infs/nans
        let mut optimizer = self.optimizer.lock().await;
        let found_infs = optimizer.unscale_gradients(scaler.scale);

        // Update scaler based on gradient statistics
        if found_infs {
            scaler.update(false);
        } else {
            scaler.update(true);
        }

        Ok(loss)
    }

    async fn apply_accumulated_gradients(&self, gradients: &[Tensor]) -> Result<()> {
        let mut optimizer = self.optimizer.lock().await;
        
        // Average gradients
        let num_gradients = gradients.len() as f64;
        for grad in gradients {
            grad.div_(num_gradients);
        }

        // Clip gradients
        if self.config.max_gradient_norm > 0.0 {
            nn::clip_grad_norm_(
                &optimizer.params,
                self.config.max_gradient_norm,
            );
        }

        // Apply gradients
        optimizer.step();
        optimizer.zero_grad();

        Ok(())
    }

    async fn evaluate(&self, model: &Arc<dyn Model>) -> Result<EvaluationMetrics> {
        let mut metrics = EvaluationMetrics::default();
        let start_time = Instant::now();

        // Evaluation logic here
        // ...

        metrics.evaluation_time = start_time.elapsed();
        Ok(metrics)
    }

    async fn save_checkpoint(
        &self,
        model: &Arc<dyn Model>,
        metrics: &TrainingMetrics,
        step: usize,
    ) -> Result<()> {
        let checkpoint = Checkpoint {
            model_state: model.parameters(),
            optimizer_state: self.optimizer.lock().await.state_dict(),
            scheduler_state: self.scheduler.as_ref().map(|s| s.lock().await.state_dict()),
            metrics: metrics.clone(),
            step,
        };

        checkpoint.save(&format!("checkpoint_{}.pt", step))?;
        Ok(())
    }

    fn compute_loss(&self, outputs: &Tensor, targets: &Tensor) -> Tensor {
        // Loss computation logic here
        outputs.cross_entropy_loss(targets, None, tch::Reduction::Mean)
    }
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub step: usize,
    pub loss_history: VecDeque<f64>,
    pub evaluation_history: Vec<EvaluationMetrics>,
    pub learning_rates: Vec<f64>,
    pub training_time: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct EvaluationMetrics {
    pub loss: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub evaluation_time: Duration,
}

struct EarlyStopper {
    config: EarlyStoppingConfig,
    best_value: f64,
    patience_counter: usize,
    best_step: usize,
}

impl EarlyStopper {
    fn new(config: &EarlyStoppingConfig) -> Self {
        Self {
            config: config.clone(),
            best_value: match config.mode {
                MonitorMode::Min => f64::INFINITY,
                MonitorMode::Max => f64::NEG_INFINITY,
            },
            patience_counter: 0,
            best_step: 0,
        }
    }

    fn should_stop(&mut self, metrics: &EvaluationMetrics) -> bool {
        let current_value = match self.config.monitor_metric.as_str() {
            "loss" => metrics.loss,
            "accuracy" => metrics.accuracy,
            "f1_score" => metrics.f1_score,
            _ => return false,
        };

        let improved = match self.config.mode {
            MonitorMode::Min => current_value < (self.best_value - self.config.min_delta),
            MonitorMode::Max => current_value > (self.best_value + self.config.min_delta),
        };

        if improved {
            self.best_value = current_value;
            self.patience_counter = 0;
        } else {
            self.patience_counter += 1;
        }

        self.patience_counter >= self.config.patience
    }
}

#[derive(Debug)]
struct Checkpoint {
    model_state: Vec<nn::Parameter>,
    optimizer_state: HashMap<String, Tensor>,
    scheduler_state: Option<HashMap<String, f64>>,
    metrics: TrainingMetrics,
    step: usize,
}

impl Checkpoint {
    fn save(&self, path: &str) -> Result<()> {
        let mut vs = nn::VarStore::new(Device::Cpu);
        // Saving logic here
        Ok(())
    }

    fn load(path: &str) -> Result<Self> {
        // Loading logic here
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::kind::Kind;

    #[tokio::test]
    async fn test_training_pipeline() {
        let config = TrainingConfig {
            batch_size: 32,
            max_steps: 100,
            gradient_accumulation_steps: 4,
            max_gradient_norm: 1.0,
            learning_rate: 1e-4,
            warmup_steps: 10,
            evaluation_steps: 20,
            checkpoint_steps: 50,
            mixed_precision: false,
            distributed: None,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 5,
                min_delta: 1e-4,
                monitor_metric: "loss".to_string(),
                mode: MonitorMode::Min,
            }),
        };

        let device_manager = Arc::new(DeviceManager::new());
        let metrics_collector = Arc::new(MetricsCollector::new());

        let pipeline = TrainingPipeline::new(
            config,
            device_manager,
            metrics_collector,
        ).await.unwrap();

        // Create dummy dataset
        let dataset = futures::stream::iter(vec![
            Ok(Tensor::zeros(&[32, 10], (Kind::Float, Device::Cpu))),
            Ok(Tensor::zeros(&[32, 10], (Kind::Float, Device::Cpu))),
        ]);

        // Create dummy model
        let model = Arc::new(DummyModel::new());

        let metrics = pipeline.train(model, dataset).await.unwrap();
        assert!(metrics.step > 0);
    }

    #[tokio::test]
    async fn test_mixed_precision_training() {
        let config = TrainingConfig {
            mixed_precision: true,
            ..TrainingConfig::default()
        };

        let device_manager = Arc::new(DeviceManager::new());
        let metrics_collector = Arc::new(MetricsCollector::new());

        let pipeline = TrainingPipeline::new(
            config,
            device_manager,
            metrics_collector,
        ).await.unwrap();

        let batch = Tensor::zeros(&[32, 10], (Kind::Float, Device::Cpu));
        let model = Arc::new(DummyModel::new());

        let loss = pipeline.train_step_mixed_precision(&model, &batch).await.unwrap();
        assert!(loss.double_value(&[]) > 0.0);
    }

    struct DummyModel {
        vs: nn::VarStore,
        linear: nn::Linear,
    }

    impl DummyModel {
        fn new() -> Self {
            let vs = nn::VarStore::new(Device::Cpu);
            let linear = nn::linear(&vs.root(), 10, 2, Default::default());
            Self { vs, linear }
        }
    }

    #[async_trait::async_trait]
    impl Model for DummyModel {
        async fn forward(&self, input: &Tensor) -> Result<Tensor> {
            Ok(self.linear.forward(input))
        }

        async fn backward(&self, gradient: &Tensor) -> Result<()> {
            gradient.backward();
            Ok(())
        }

        async fn save(&self, path: &Path) -> Result<()> {
            self.vs.save(path)?;
            Ok(())
        }

        async fn load(&mut self, path: &Path) -> Result<()> {
            self.vs.load(path)?;
            Ok(())
        }

        fn parameters(&self) -> Vec<nn::Parameter> {
            self.vs.trainable_variables()
        }

        fn architecture(&self) -> ModelArchitecture {
            ModelArchitecture::Custom(CustomConfig {
                name: "Dummy".to_string(),
                params: HashMap::new(),
            })
        }
    }

    #[tokio::test]
    async fn test_gradient_accumulation() {
        let config = TrainingConfig {
            gradient_accumulation_steps: 4,
            batch_size: 8,
            ..TrainingConfig::default()
        };

        let device_manager = Arc::new(DeviceManager::new());
        let metrics_collector = Arc::new(MetricsCollector::new());
        let pipeline = TrainingPipeline::new(
            config,
            device_manager,
            metrics_collector,
        ).await.unwrap();

        let mut gradients = Vec::new();
        let model = Arc::new(DummyModel::new());
        
        for _ in 0..4 {
            let batch = Tensor::zeros(&[8, 10], (Kind::Float, Device::Cpu));
            let loss = pipeline.train_step(&model, &batch).await.unwrap();
            gradients.push(loss.backward());
        }

        pipeline.apply_accumulated_gradients(&gradients).await.unwrap();
    }

    #[tokio::test]
    async fn test_early_stopping() {
        let config = EarlyStoppingConfig {
            patience: 3,
            min_delta: 1e-4,
            monitor_metric: "loss".to_string(),
            mode: MonitorMode::Min,
        };

        let mut stopper = EarlyStopper::new(&config);

        // Test improving metrics
        let metrics1 = EvaluationMetrics {
            loss: 1.0,
            ..Default::default()
        };
        assert!(!stopper.should_stop(&metrics1));

        let metrics2 = EvaluationMetrics {
            loss: 0.8,
            ..Default::default()
        };
        assert!(!stopper.should_stop(&metrics2));

        // Test stagnating metrics
        for _ in 0..3 {
            let metrics = EvaluationMetrics {
                loss: 0.8,
                ..Default::default()
            };
            if stopper.should_stop(&metrics) {
                break;
            }
        }
        assert!(stopper.should_stop(&metrics2));
    }

    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = tempfile::tempdir().unwrap();
        let checkpoint_path = temp_dir.path().join("checkpoint.pt");

        let model = DummyModel::new();
        let metrics = TrainingMetrics {
            step: 100,
            loss_history: VecDeque::from(vec![0.5, 0.4, 0.3]),
            evaluation_history: vec![EvaluationMetrics::default()],
            learning_rates: vec![1e-4],
            training_time: Duration::from_secs(60),
        };

        let checkpoint = Checkpoint {
            model_state: model.parameters(),
            optimizer_state: HashMap::new(),
            scheduler_state: None,
            metrics: metrics.clone(),
            step: 100,
        };

        checkpoint.save(checkpoint_path.to_str().unwrap()).unwrap();
        assert!(checkpoint_path.exists());
    }

    #[tokio::test]
    async fn test_learning_rate_scheduler() {
        let scheduler = LRScheduler::new(1e-3, 1000, 100);
        let mut current_lr = scheduler.get_lr();
        
        // Test warmup phase
        for _ in 0..100 {
            scheduler.step();
            let new_lr = scheduler.get_lr();
            assert!(new_lr > current_lr);
            current_lr = new_lr;
        }

        // Test decay phase
        for _ in 100..1000 {
            scheduler.step();
            let new_lr = scheduler.get_lr();
            assert!(new_lr <= current_lr);
            current_lr = new_lr;
        }
    }

    #[tokio::test]
    async fn test_gradient_scaler() {
        let mut scaler = GradientScaler::new();
        
        // Test successful steps
        for _ in 0..100 {
            scaler.update(true);
        }
        assert!(scaler.scale > 1.0);

        // Test failed steps
        scaler.update(false);
        let scale_after_failure = scaler.scale;
        assert!(scale_after_failure < scaler.scale);
    }

    proptest::proptest! {
        #[test]
        fn test_loss_computation(
            batch_size in 1..32i64,
            features in 1..100i64
        ) {
            let outputs = Tensor::zeros(&[batch_size, features], (Kind::Float, Device::Cpu));
            let targets = Tensor::zeros(&[batch_size], (Kind::Int64, Device::Cpu));
            
            let pipeline = TrainingPipeline::new(
                TrainingConfig::default(),
                Arc::new(DeviceManager::new()),
                Arc::new(MetricsCollector::new()),
            ).await.unwrap();

            let loss = pipeline.compute_loss(&outputs, &targets);
            prop_assert!(loss.dim() == 0);
            prop_assert!(loss.requires_grad());
        }
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();
        
        // Test metric updates
        for i in 0..10 {
            metrics.update(1.0 / (i + 1) as f64, i);
        }
        
        assert_eq!(metrics.loss_history.len(), 10);
        assert!(metrics.loss_history.iter().zip(metrics.loss_history.iter().skip(1))
            .all(|(a, b)| a >= b));
    }
}