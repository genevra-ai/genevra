use std::{
    sync::Arc,
    collections::{HashMap, BTreeMap},
    time::{Duration, Instant},
};

use anyhow::{Result, Context};
use metrics::{counter, gauge, histogram};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use tokio::sync::broadcast;
use ringbuffer::{RingBuffer, ConstGenericRingBuffer};

const METRICS_BUFFER_SIZE: usize = 1000;
const AGGREGATION_WINDOW: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub inference_metrics: InferenceMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub resource_metrics: ResourceMetrics,
    pub error_metrics: ErrorMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_latency: Duration,
    pub p50_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub throughput: f64,
    pub batch_statistics: BatchStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    pub average_batch_size: f64,
    pub max_batch_size: usize,
    pub min_batch_size: usize,
    pub batch_size_distribution: HashMap<usize, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cpu_utilization: f64,
    pub memory_usage: usize,
    pub gpu_utilization: Option<f64>,
    pub gpu_memory_usage: Option<usize>,
    pub disk_io: DiskIOMetrics,
    pub network_io: NetworkIOMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub memory_allocation: MemoryMetrics,
    pub thread_utilization: ThreadMetrics,
    pub cache_statistics: CacheMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    pub error_count: u64,
    pub error_types: HashMap<String, u64>,
    pub error_rates: HashMap<String, f64>,
    pub last_error_timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOMetrics {
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub read_ops: u64,
    pub write_ops: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIOMetrics {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub active_connections: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub current_allocated: usize,
    pub allocation_histogram: Vec<(usize, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadMetrics {
    pub active_threads: usize,
    pub thread_pool_size: usize,
    pub queue_depth: usize,
    pub task_completion_times: Vec<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size: usize,
    pub capacity: usize,
}

pub struct MetricsCollector {
    metrics_buffer: Arc<RwLock<ConstGenericRingBuffer<ModelMetrics, METRICS_BUFFER_SIZE>>>,
    current_window: Arc<RwLock<BTreeMap<String, WindowedMetrics>>>,
    aggregated_metrics: Arc<RwLock<HashMap<String, AggregatedMetrics>>>,
    notification_tx: broadcast::Sender<MetricsUpdate>,
}

#[derive(Debug)]
struct WindowedMetrics {
    start_time: Instant,
    latencies: Vec<Duration>,
    request_count: u64,
    error_count: u64,
    batch_sizes: Vec<usize>,
    resource_samples: Vec<ResourceSample>,
}

#[derive(Debug)]
struct ResourceSample {
    timestamp: Instant,
    cpu_usage: f64,
    memory_usage: usize,
    gpu_usage: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct AggregatedMetrics {
    window_duration: Duration,
    request_rate: f64,
    error_rate: f64,
    latency_stats: LatencyStatistics,
    resource_usage: ResourceUsageStatistics,
}

#[derive(Debug, Clone, Serialize)]
struct LatencyStatistics {
    mean: Duration,
    median: Duration,
    p95: Duration,
    p99: Duration,
    std_dev: Duration,
}

#[derive(Debug, Clone, Serialize)]
struct ResourceUsageStatistics {
    cpu_usage_mean: f64,
    memory_usage_mean: usize,
    gpu_usage_mean: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum MetricsUpdate {
    NewDataPoint(String, ModelMetrics),
    WindowCompleted(String, AggregatedMetrics),
    Alert(MetricsAlert),
}

#[derive(Debug, Clone)]
pub enum MetricsAlert {
    HighLatency { model_id: String, latency: Duration },
    HighErrorRate { model_id: String, rate: f64 },
    ResourceExhaustion { resource: String, usage: f64 },
}

impl MetricsCollector {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1000);
        
        Self {
            metrics_buffer: Arc::new(RwLock::new(ConstGenericRingBuffer::new())),
            current_window: Arc::new(RwLock::new(BTreeMap::new())),
            aggregated_metrics: Arc::new(RwLock::new(HashMap::new())),
            notification_tx: tx,
        }
    }

    pub fn record_inference(&self, model_id: &str, latency: Duration, success: bool) {
        let mut window = self.current_window.write();
        let metrics = window.entry(model_id.to_string())
            .or_insert_with(|| WindowedMetrics::new(Instant::now()));

        metrics.latencies.push(latency);
        metrics.request_count += 1;
        if !success {
            metrics.error_count += 1;
        }

        // Record prometheus metrics
        histogram!("model_inference_latency", latency);
        counter!("model_inference_requests_total", 1);
        if !success {
            counter!("model_inference_errors_total", 1);
        }
    }

    pub fn record_batch(&self, model_id: &str, batch_size: usize) {
        let mut window = self.current_window.write();
        let metrics = window.entry(model_id.to_string())
            .or_insert_with(|| WindowedMetrics::new(Instant::now()));

        metrics.batch_sizes.push(batch_size);
        gauge!("model_batch_size", batch_size as f64);
    }

    pub fn record_resource_usage(
        &self,
        model_id: &str,
        cpu_usage: f64,
        memory_usage: usize,
        gpu_usage: Option<f64>,
    ) {
        let mut window = self.current_window.write();
        let metrics = window.entry(model_id.to_string())
            .or_insert_with(|| WindowedMetrics::new(Instant::now()));

        let sample = ResourceSample {
            timestamp: Instant::now(),
            cpu_usage,
            memory_usage,
            gpu_usage,
        };

        metrics.resource_samples.push(sample);

        // Record prometheus metrics
        gauge!("model_cpu_usage", cpu_usage);
        gauge!("model_memory_usage", memory_usage as f64);
        if let Some(gpu) = gpu_usage {
            gauge!("model_gpu_usage", gpu);
        }
    }

    pub async fn aggregate_metrics(&self) -> Result<()> {
        let mut window = self.current_window.write();
        let now = Instant::now();

        for (model_id, metrics) in window.iter() {
            if now.duration_since(metrics.start_time) >= AGGREGATION_WINDOW {
                let aggregated = self.compute_aggregated_metrics(metrics);
                
                self.aggregated_metrics.write()
                    .insert(model_id.clone(), aggregated.clone());

                // Notify subscribers
                let _ = self.notification_tx.send(MetricsUpdate::WindowCompleted(
                    model_id.clone(),
                    aggregated,
                ));

                // Check for alerts
                self.check_alerts(model_id, metrics);
            }
        }

        // Remove completed windows
        window.retain(|_, m| now.duration_since(m.start_time) < AGGREGATION_WINDOW);

        Ok(())
    }

    fn compute_aggregated_metrics(&self, window: &WindowedMetrics) -> AggregatedMetrics {
        let mut latencies = window.latencies.clone();
        latencies.sort_unstable();

        let request_rate = window.request_count as f64 / AGGREGATION_WINDOW.as_secs_f64();
        let error_rate = window.error_count as f64 / window.request_count as f64;

        AggregatedMetrics {
            window_duration: AGGREGATION_WINDOW,
            request_rate,
            error_rate,
            latency_stats: self.compute_latency_statistics(&latencies),
            resource_usage: self.compute_resource_statistics(&window.resource_samples),
        }
    }

    fn compute_latency_statistics(&self, latencies: &[Duration]) -> LatencyStatistics {
        if latencies.is_empty() {
            return LatencyStatistics::default();
        }

        let mean = latencies.iter().sum::<Duration>() / latencies.len() as u32;
        let median = latencies[latencies.len() / 2];
        let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

        let variance = latencies.iter()
            .map(|&lat| {
                let diff = lat.as_secs_f64() - mean.as_secs_f64();
                diff * diff
            })
            .sum::<f64>() / latencies.len() as f64;

        let std_dev = Duration::from_secs_f64(variance.sqrt());

        LatencyStatistics {
            mean,
            median,
            p95,
            p99,
            std_dev,
        }
    }

    fn compute_resource_statistics(&self, samples: &[ResourceSample]) -> ResourceUsageStatistics {
        if samples.is_empty() {
            return ResourceUsageStatistics::default();
        }

        let cpu_usage_mean = samples.iter()
            .map(|s| s.cpu_usage)
            .sum::<f64>() / samples.len() as f64;

        let memory_usage_mean = samples.iter()
            .map(|s| s.memory_usage)
            .sum::<usize>() / samples.len();

        let gpu_usage_mean = if samples.iter().any(|s| s.gpu_usage.is_some()) {
            let sum = samples.iter()
                .filter_map(|s| s.gpu_usage)
                .sum::<f64>();
            let count = samples.iter()
                .filter(|s| s.gpu_usage.is_some())
                .count();
            Some(sum / count as f64)
        } else {
            None
        };

        ResourceUsageStatistics {
            cpu_usage_mean,
            memory_usage_mean,
            gpu_usage_mean,
        }
    }

    fn check_alerts(&self, model_id: &str, metrics: &WindowedMetrics) {
        // Check latency threshold
        if let Some(p99) = metrics.latencies.iter().max() {
            if p99 > &Duration::from_millis(500) {
                let _ = self.notification_tx.send(MetricsUpdate::Alert(
                    MetricsAlert::HighLatency {
                        model_id: model_id.to_string(),
                        latency: *p99,
                    }
                ));
            }
        }

        // Check error rate
        let error_rate = metrics.error_count as f64 / metrics.request_count as f64;
        if error_rate > 0.05 {
            let _ = self.notification_tx.send(MetricsUpdate::Alert(
                MetricsAlert::HighErrorRate {
                    model_id: model_id.to_string(),
                    rate: error_rate,
                }
            ));
        }

        // Check resource usage
        if let Some(last_sample) = metrics.resource_samples.last() {
            if last_sample.cpu_usage > 90.0 {
                let _ = self.notification_tx.send(MetricsUpdate::Alert(
                    MetricsAlert::ResourceExhaustion {
                        resource: "CPU".to_string(),
                        usage: last_sample.cpu_usage,
                    }
                ));
            }
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<MetricsUpdate> {
        self.notification_tx.subscribe()
    }
}

impl WindowedMetrics {
    fn new(start_time: Instant) -> Self {
        Self {
            start_time,
            latencies: Vec::new(),
            request_count: 0,
            error_count: 0,
            batch_sizes: Vec::new(),
            resource_samples: Vec::new(),
        }
    }
}

impl Default for LatencyStatistics {
    fn default() -> Self {
        Self {
            mean: Duration::default(),
            median: Duration::default(),
            p95: Duration::default(),
            p99: Duration::default(),
            std_dev: Duration::default(),
        }
    }
}

impl Default for ResourceUsageStatistics {
    fn default() -> Self {
        Self {
            cpu_usage_mean: 0.0,
            memory_usage_mean: 0,
            gpu_usage_mean: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;
    use approx::assert_relative_eq;

    #[test]
    fn test_metrics_recording() {
        let collector = MetricsCollector::new();
        
        collector.record_inference("model1", Duration::from_millis(100), true);
        collector.record_inference("model1", Duration::from_millis(150), true);
        collector.record_inference("model1", Duration::from_millis(200), false);

        let window = collector.current_window.read();
        let metrics = window.get("model1").unwrap();

        assert_eq!(metrics.request_count, 3);
        assert_eq!(metrics.error_count, 1);
        assert_eq!(metrics.latencies.len(), 3);
    }

    #[tokio::test]
    async fn test_metrics_aggregation() {
        let collector = MetricsCollector::new();

        // Record some metrics
        for i in 0..100 {
            collector.record_inference(
                "model1",
                Duration::from_millis(100 + i),
                i % 10 != 0,
            );
            collector.record_resource_usage(
                "model1",
                50.0 + (i as f64 / 10.0),
                1000 + i * 100,
                Some(30.0 + (i as f64 / 5.0)),
            );
            collector.record_batch("model1", 16 + (i % 16));
        }

        // Wait for window to complete
        sleep(AGGREGATION_WINDOW + Duration::from_millis(100)).await;
        collector.aggregate_metrics().await.unwrap();

        let aggregated = collector.aggregated_metrics.read();
        let metrics = aggregated.get("model1").unwrap();

        assert!(metrics.request_rate > 0.0);
        assert_relative_eq!(metrics.error_rate, 0.1, epsilon = 0.01);
    }

    #[test]
    fn test_latency_statistics() {
        let collector = MetricsCollector::new();
        let latencies = vec![
            Duration::from_millis(100),
            Duration::from_millis(150),
            Duration::from_millis(200),
            Duration::from_millis(250),
            Duration::from_millis(300),
        ];

        let stats = collector.compute_latency_statistics(&latencies);

        assert_eq!(stats.mean, Duration::from_millis(200));
        assert_eq!(stats.median, Duration::from_millis(200));
        assert_eq!(stats.p95, Duration::from_millis(300));
        assert_eq!(stats.p99, Duration::from_millis(300));
    }

    #[test]
    fn test_resource_statistics() {
        let collector = MetricsCollector::new();
        let samples = vec![
            ResourceSample {
                timestamp: Instant::now(),
                cpu_usage: 50.0,
                memory_usage: 1000,
                gpu_usage: Some(30.0),
            },
            ResourceSample {
                timestamp: Instant::now(),
                cpu_usage: 60.0,
                memory_usage: 1200,
                gpu_usage: Some(35.0),
            },
        ];

        let stats = collector.compute_resource_statistics(&samples);

        assert_relative_eq!(stats.cpu_usage_mean, 55.0);
        assert_eq!(stats.memory_usage_mean, 1100);
        assert_relative_eq!(stats.gpu_usage_mean.unwrap(), 32.5);
    }

    #[tokio::test]
    async fn test_metrics_alerts() {
        let collector = MetricsCollector::new();
        let mut receiver = collector.subscribe();

        // Trigger high latency alert
        collector.record_inference(
            "model1",
            Duration::from_millis(1000),
            true,
        );

        // Trigger error rate alert
        for _ in 0..10 {
            collector.record_inference(
                "model1",
                Duration::from_millis(100),
                false,
            );
        }

        // Trigger resource exhaustion alert
        collector.record_resource_usage(
            "model1",
            95.0,
            1000,
            Some(80.0),
        );

        collector.aggregate_metrics().await.unwrap();

        let mut alerts = Vec::new();
        while let Ok(update) = receiver.try_recv() {
            if let MetricsUpdate::Alert(alert) = update {
                alerts.push(alert);
            }
        }

        assert!(alerts.iter().any(|a| matches!(a, MetricsAlert::HighLatency { .. })));
        assert!(alerts.iter().any(|a| matches!(a, MetricsAlert::HighErrorRate { .. })));
        assert!(alerts.iter().any(|a| matches!(a, MetricsAlert::ResourceExhaustion { .. })));
    }

    #[test]
    fn test_batch_statistics() {
        let collector = MetricsCollector::new();

        for i in 0..100 {
            collector.record_batch("model1", 16 + (i % 16));
        }

        let window = collector.current_window.read();
        let metrics = window.get("model1").unwrap();

        assert_eq!(metrics.batch_sizes.len(), 100);
        assert!(*metrics.batch_sizes.iter().max().unwrap() <= 32);
        assert!(*metrics.batch_sizes.iter().min().unwrap() >= 16);
    }

    #[tokio::test]
    async fn test_metrics_window_rotation() {
        let collector = MetricsCollector::new();

        // Record metrics in first window
        collector.record_inference("model1", Duration::from_millis(100), true);
        
        // Wait for window to complete
        sleep(AGGREGATION_WINDOW + Duration::from_millis(100)).await;
        collector.aggregate_metrics().await.unwrap();

        // Record metrics in new window
        collector.record_inference("model1", Duration::from_millis(150), true);

        let window = collector.current_window.read();
        let metrics = window.get("model1").unwrap();

        assert_eq!(metrics.request_count, 1);
        assert_eq!(metrics.latencies.len(), 1);
    }

    #[test]
    fn test_empty_metrics() {
        let collector = MetricsCollector::new();
        
        let latency_stats = collector.compute_latency_statistics(&[]);
        assert_eq!(latency_stats.mean, Duration::default());
        assert_eq!(latency_stats.median, Duration::default());

        let resource_stats = collector.compute_resource_statistics(&[]);
        assert_eq!(resource_stats.cpu_usage_mean, 0.0);
        assert_eq!(resource_stats.memory_usage_mean, 0);
        assert!(resource_stats.gpu_usage_mean.is_none());
    }

    proptest::proptest! {
        #[test]
        fn test_random_metrics(
            latencies in proptest::collection::vec(0u64..1000, 1..100),
            error_rate in 0.0..1.0f64
        ) {
            let collector = MetricsCollector::new();
            
            for latency in latencies.iter() {
                collector.record_inference(
                    "model1",
                    Duration::from_millis(*latency),
                    rand::random::<f64>() > error_rate,
                );
            }

            let window = collector.current_window.read();
            let metrics = window.get("model1").unwrap();
            
            prop_assert!(metrics.request_count == latencies.len() as u64);
            prop_assert!(metrics.latencies.len() == latencies.len());
        }
    }
}