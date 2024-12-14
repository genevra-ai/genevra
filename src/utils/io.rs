use std::{
    sync::Arc,
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};

use anyhow::{Result, Context, bail};
use tokio::sync::{RwLock, Semaphore, mpsc};
use async_trait::async_trait;
use metrics::{counter, gauge, histogram};

pub mod io;
pub mod compression;
pub mod validation;
pub mod visualization;

const MAX_QUEUE_SIZE: usize = 10_000;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug)]
pub struct Queue<T> {
    items: VecDeque<T>,
    max_size: usize,
    metrics: QueueMetrics,
}

#[derive(Debug)]
pub struct BatchProcessor<T, R> {
    queue: Arc<RwLock<Queue<T>>>,
    processor: Arc<dyn BatchProcessorBackend<T, R> + Send + Sync>,
    batch_size: usize,
    semaphore: Arc<Semaphore>,
}

#[derive(Debug)]
struct QueueMetrics {
    enqueued: u64,
    dequeued: u64,
    dropped: u64,
    processing_times: Vec<Duration>,
}

#[async_trait]
pub trait BatchProcessorBackend<T, R>: Send + Sync {
    async fn process_batch(&self, items: Vec<T>) -> Result<Vec<R>>;
}

impl<T> Queue<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(max_size),
            max_size,
            metrics: QueueMetrics {
                enqueued: 0,
                dequeued: 0,
                dropped: 0,
                processing_times: Vec::new(),
            },
        }
    }

    pub fn push(&mut self, item: T) -> bool {
        if self.items.len() >= self.max_size {
            self.metrics.dropped += 1;
            gauge!("queue_dropped_items", self.metrics.dropped as f64);
            false
        } else {
            self.items.push_back(item);
            self.metrics.enqueued += 1;
            gauge!("queue_size", self.items.len() as f64);
            counter!("queue_enqueued_items", 1);
            true
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let item = self.items.pop_front();
        if item.is_some() {
            self.metrics.dequeued += 1;
            gauge!("queue_size", self.items.len() as f64);
            counter!("queue_dequeued_items", 1);
        }
        item
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn clear(&mut self) {
        self.items.clear();
        gauge!("queue_size", 0.0);
    }

    pub fn metrics(&self) -> &QueueMetrics {
        &self.metrics
    }
}

impl<T, R> BatchProcessor<T, R>
where
    T: Send + Sync + 'static,
    R: Send + Sync + 'static,
{
    pub fn new(
        processor: Arc<dyn BatchProcessorBackend<T, R> + Send + Sync>,
        batch_size: usize,
        max_concurrent: usize,
    ) -> Self {
        Self {
            queue: Arc::new(RwLock::new(Queue::new(MAX_QUEUE_SIZE))),
            processor,
            batch_size,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    pub async fn submit(&self, item: T) -> Result<()> {
        let mut queue = self.queue.write().await;
        if !queue.push(item) {
            bail!("Queue is full");
        }
        
        if queue.len() >= self.batch_size {
            self.process_batch().await?;
        }
        
        Ok(())
    }

    pub async fn process_batch(&self) -> Result<Vec<R>> {
        let _permit = self.semaphore.acquire().await?;
        let start = Instant::now();

        let mut items = Vec::with_capacity(self.batch_size);
        {
            let mut queue = self.queue.write().await;
            while let Some(item) = queue.pop() {
                items.push(item);
                if items.len() >= self.batch_size {
                    break;
                }
            }
        }

        if items.is_empty() {
            return Ok(Vec::new());
        }

        let results = self.processor.process_batch(items).await?;
        
        let duration = start.elapsed();
        histogram!("batch_processing_time", duration);
        {
            let mut queue = self.queue.write().await;
            queue.metrics.processing_times.push(duration);
        }

        Ok(results)
    }

    pub async fn flush(&self) -> Result<Vec<R>> {
        let mut all_results = Vec::new();
        while !self.queue.read().await.is_empty() {
            let results = self.process_batch().await?;
            all_results.extend(results);
        }
        Ok(all_results)
    }
}

pub struct ProgressTracker {
    total: u64,
    current: u64,
    start_time: Instant,
    updates: mpsc::Sender<ProgressUpdate>,
}

#[derive(Debug)]
pub struct ProgressUpdate {
    pub current: u64,
    pub total: u64,
    pub elapsed: Duration,
    pub estimated_remaining: Duration,
}

impl ProgressTracker {
    pub fn new(total: u64) -> (Self, mpsc::Receiver<ProgressUpdate>) {
        let (tx, rx) = mpsc::channel(100);
        (
            Self {
                total,
                current: 0,
                start_time: Instant::now(),
                updates: tx,
            },
            rx
        )
    }

    pub fn increment(&mut self, amount: u64) {
        self.current += amount;
        let elapsed = self.start_time.elapsed();
        
        if self.current > 0 {
            let items_per_sec = self.current as f64 / elapsed.as_secs_f64();
            let remaining_items = self.total - self.current;
            let estimated_remaining = Duration::from_secs_f64(remaining_items as f64 / items_per_sec);

            let _ = self.updates.try_send(ProgressUpdate {
                current: self.current,
                total: self.total,
                elapsed,
                estimated_remaining,
            });
        }
    }

    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }
}

pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

pub fn format_size(size: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut size = size as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_queue_operations() {
        let mut queue = Queue::new(2);
        
        assert!(queue.push(1));
        assert!(queue.push(2));
        assert!(!queue.push(3)); // Should fail, queue is full
        
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), None);
    }

    #[derive(Debug)]
    struct TestProcessor;

    #[async_trait]
    impl BatchProcessorBackend<i32, i32> for TestProcessor {
        async fn process_batch(&self, items: Vec<i32>) -> Result<Vec<i32>> {
            Ok(items.into_iter().map(|x| x * 2).collect())
        }
    }

    #[tokio::test]
    async fn test_batch_processor() {
        let processor = Arc::new(TestProcessor);
        let batch_processor = BatchProcessor::new(processor, 2, 1);

        batch_processor.submit(1).await.unwrap();
        batch_processor.submit(2).await.unwrap();
        batch_processor.submit(3).await.unwrap();

        let results = batch_processor.flush().await.unwrap();
        assert_eq!(results, vec![2, 4, 6]);
    }

    #[test]
    fn test_progress_tracker() {
        let (mut tracker, mut rx) = ProgressTracker::new(100);
        
        tracker.increment(50);
        let update = rx.try_recv().unwrap();
        assert_eq!(update.current, 50);
        assert_eq!(update.total, 100);
        
        tracker.increment(50);
        assert!(tracker.is_complete());
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3665)), "1h 1m 5s");
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500.00 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
    }

    proptest::proptest! {
        #[test]
        fn test_queue_push_pop(items in proptest::collection::vec(0i32..100, 0..10)) {
            let mut queue = Queue::new(10);
            let mut expected = VecDeque::new();

            for item in items {
                if queue.push(item) {
                    expected.push_back(item);
                }
            }

            while let Some(expected_item) = expected.pop_front() {
                assert_eq!(queue.pop(), Some(expected_item));
            }

            assert!(queue.is_empty());
        }
    }
}