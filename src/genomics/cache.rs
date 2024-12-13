use std::{
    collections::{HashMap, BTreeMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
    path::PathBuf,
    io,
};

use anyhow::{Result, Context, bail};
use dashmap::DashMap;
use futures::{Stream, StreamExt};
use lru::LruCache;
use parking_lot::{RwLock, Mutex};
use serde::{Serialize, Deserialize};
use tokio::{fs, sync::broadcast};
use metrics::{counter, gauge, histogram};

const DEFAULT_CACHE_SIZE: usize = 1024 * 1024 * 1024; // 1GB
const EVICTION_THRESHOLD: f64 = 0.9;
const CLEANUP_INTERVAL: Duration = Duration::from_secs(300);

#[derive(Debug)]
pub struct Cache {
    memory_store: Arc<DashMap<String, CacheEntry>>,
    disk_store: Arc<RwLock<DiskStore>>,
    lru_index: Arc<Mutex<LruCache<String, CacheMetadata>>>,
    config: CacheConfig,
    metrics: Arc<CacheMetrics>,
    eviction_tx: broadcast::Sender<EvictionEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub max_memory_size: usize,
    pub max_disk_size: usize,
    pub eviction_policy: EvictionPolicy,
    pub compression_level: CompressionLevel,
    pub persistence_path: Option<PathBuf>,
    pub cache_validation: CacheValidation,
    pub ttl: Option<Duration>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    ARC,
    Custom(u32),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionLevel {
    None,
    Fast,
    Default,
    Best,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheValidation {
    None,
    Checksum,
    StrongConsistency,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Arc<Vec<u8>>,
    metadata: CacheMetadata,
    compression_info: Option<CompressionInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheMetadata {
    key: String,
    size: usize,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    checksum: Option<String>,
    ttl: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompressionInfo {
    algorithm: CompressionLevel,
    original_size: usize,
    compressed_size: usize,
    compression_ratio: f64,
}

#[derive(Debug)]
struct DiskStore {
    base_path: PathBuf,
    index: BTreeMap<String, DiskEntryMetadata>,
    current_size: usize,
    max_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskEntryMetadata {
    path: PathBuf,
    size: usize,
    checksum: String,
    created_at: Instant,
}

#[derive(Debug, Clone)]
struct EvictionEvent {
    key: String,
    reason: EvictionReason,
    size: usize,
}

#[derive(Debug, Clone, Copy)]
enum EvictionReason {
    MemoryPressure,
    TTLExpired,
    Manual,
    Invalidated,
}

#[derive(Debug, Default)]
pub struct CacheMetrics {
    hits: metrics::Counter,
    misses: metrics::Counter,
    evictions: metrics::Counter,
    memory_usage: metrics::Gauge,
    disk_usage: metrics::Gauge,
    compression_ratio: metrics::Histogram,
    operation_latency: metrics::Histogram,
}

impl Cache {
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let memory_store = Arc::new(DashMap::new());
        let lru_index = Arc::new(Mutex::new(LruCache::new(
            config.max_memory_size / std::mem::size_of::<CacheMetadata>()
        )));
        
        let disk_store = if let Some(path) = &config.persistence_path {
            fs::create_dir_all(path).await?;
            Arc::new(RwLock::new(DiskStore::new(path.clone(), config.max_disk_size)))
        } else {
            Arc::new(RwLock::new(DiskStore::new_in_memory(config.max_disk_size)))
        };

        let (eviction_tx, _) = broadcast::channel(1024);
        let metrics = Arc::new(CacheMetrics::new());

        let cache = Self {
            memory_store,
            disk_store,
            lru_index,
            config,
            metrics,
            eviction_tx,
        };

        // Start background maintenance tasks
        cache.start_maintenance_tasks();

        Ok(cache)
    }

    pub async fn get<T: for<'de> serde::Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        let start = Instant::now();

        let result = if let Some(entry) = self.memory_store.get(key) {
            // Update access metadata
            self.update_access_metadata(key, &entry).await?;
            
            // Decompress if necessary
            let data = if let Some(compression_info) = &entry.metadata.compression_info {
                self.decompress_data(&entry.data, compression_info)?
            } else {
                entry.data.to_vec()
            };

            // Validate if required
            if self.validate_entry(&entry).await? {
                counter!(self.metrics.hits, 1);
                Some(bincode::deserialize(&data)?)
            } else {
                self.invalidate(key).await?;
                None
            }
        } else if let Some(disk_entry) = self.disk_store.read().get_entry(key).await? {
            // Load from disk and cache in memory
            let data = self.load_from_disk(&disk_entry).await?;
            self.cache_in_memory(key.to_string(), data.clone()).await?;
            
            counter!(self.metrics.hits, 1);
            Some(bincode::deserialize(&data)?)
        } else {
            counter!(self.metrics.misses, 1);
            None
        };

        histogram!(self.metrics.operation_latency, start.elapsed());
        Ok(result)
    }

    pub async fn set<T: Serialize>(&self, key: String, value: T, ttl: Option<Duration>) -> Result<()> {
        let start = Instant::now();

        // Serialize and potentially compress the data
        let data = bincode::serialize(&value)?;
        let (compressed_data, compression_info) = self.compress_data(&data)?;

        // Create cache entry
        let entry = CacheEntry {
            data: Arc::new(compressed_data),
            metadata: CacheMetadata {
                key: key.clone(),
                size: data.len(),
                created_at: Instant::now(),
                last_accessed: Instant::now(),
                access_count: 0,
                checksum: Some(self.calculate_checksum(&data)),
                ttl,
            },
            compression_info: Some(compression_info),
        };

        // Check if we need to evict entries
        self.ensure_capacity(entry.metadata.size).await?;

        // Store in memory and update indexes
        self.memory_store.insert(key.clone(), entry.clone());
        self.lru_index.lock().put(key.clone(), entry.metadata.clone());

        // Persist to disk if configured
        if let Some(_) = &self.config.persistence_path {
            self.disk_store.write().store_entry(&key, &entry).await?;
        }

        // Update metrics
        gauge!(self.metrics.memory_usage, self.get_memory_usage() as f64);
        histogram!(self.metrics.operation_latency, start.elapsed());
        
        Ok(())
    }

    pub async fn remove(&self, key: &str) -> Result<()> {
        let start = Instant::now();

        if let Some(entry) = self.memory_store.remove(key) {
            self.lru_index.lock().pop(&key);
            
            if let Some(_) = &self.config.persistence_path {
                self.disk_store.write().remove_entry(key).await?;
            }

            self.eviction_tx.send(EvictionEvent {
                key: key.to_string(),
                reason: EvictionReason::Manual,
                size: entry.1.metadata.size,
            })?;
        }

        histogram!(self.metrics.operation_latency, start.elapsed());
        Ok(())
    }

    pub async fn clear(&self) -> Result<()> {
        self.memory_store.clear();
        self.lru_index.lock().clear();
        self.disk_store.write().clear().await?;

        gauge!(self.metrics.memory_usage, 0.0);
        gauge!(self.metrics.disk_usage, 0.0);
        
        Ok(())
    }

    async fn ensure_capacity(&self, required_size: usize) -> Result<()> {
        let current_usage = self.get_memory_usage();
        let max_size = self.config.max_memory_size;

        if current_usage + required_size > max_size {
            let target_size = (max_size as f64 * EVICTION_THRESHOLD) as usize;
            let size_to_free = current_usage + required_size - target_size;
            
            self.evict_entries(size_to_free).await?;
        }

        Ok(())
    }

    async fn evict_entries(&self, size_to_free: usize) -> Result<()> {
        let mut freed_size = 0;
        let mut evicted_keys = Vec::new();

        match self.config.eviction_policy {
            EvictionPolicy::LRU => {
                let mut lru = self.lru_index.lock();
                while freed_size < size_to_free {
                    if let Some((key, metadata)) = lru.pop_lru() {
                        freed_size += metadata.size;
                        evicted_keys.push(key);
                    } else {
                        break;
                    }
                }
            }
            EvictionPolicy::LFU => {
                // Implement LFU eviction logic
                let entries: Vec<_> = self.memory_store.iter()
                    .map(|entry| (entry.key().clone(), entry.value().metadata.access_count))
                    .collect();

                for (key, _) in entries.iter()
                    .sorted_by_key(|(_, count)| *count)
                    .take_while(|_| freed_size < size_to_free) {
                    if let Some(entry) = self.memory_store.remove(key) {
                        freed_size += entry.1.metadata.size;
                        evicted_keys.push(key.clone());
                    }
                }
            }
            _ => {
                // Implement other eviction policies
            }
        }

        // Process evicted entries
        for key in evicted_keys {
            if let Some(_) = &self.config.persistence_path {
                // Persist to disk before evicting from memory
                if let Some(entry) = self.memory_store.get(&key) {
                    self.disk_store.write().store_entry(&key, &entry).await?;
                }
            }
            
            self.memory_store.remove(&key);
            counter!(self.metrics.evictions, 1);
            
            self.eviction_tx.send(EvictionEvent {
                key,
                reason: EvictionReason::MemoryPressure,
                size: 0, // Size already counted in freed_size
            })?;
        }

        Ok(())
    }

    fn start_maintenance_tasks(&self) {
        let memory_store = Arc::clone(&self.memory_store);
        let disk_store = Arc::clone(&self.disk_store);
        let config = self.config.clone();
        let metrics = Arc::clone(&self.metrics);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(CLEANUP_INTERVAL);
            loop {
                interval.tick().await;
                
                // Cleanup expired entries
                let expired: Vec<_> = memory_store.iter()
                    .filter(|entry| {
                        if let Some(ttl) = entry.value().metadata.ttl {
                            entry.value().metadata.created_at.elapsed() > ttl
                        } else {
                            false
                        }
                    })
                    .map(|entry| entry.key().clone())
                    .collect();

                for key in expired {
                    memory_store.remove(&key);
                    if let Some(_) = &config.persistence_path {
                        disk_store.write().remove_entry(&key).await.ok();
                    }
                }

                // Update metrics
                gauge!(metrics.memory_usage, memory_store.len() as f64);
                if let Some(_) = &config.persistence_path {
                    gauge!(metrics.disk_usage, disk_store.read().current_size as f64);
                }
            }
        });
    }

    async fn validate_entry(&self, entry: &CacheEntry) -> Result<bool> {
        match self.config.cache_validation {
            CacheValidation::None => Ok(true),
            CacheValidation::Checksum => {
                if let Some(stored_checksum) = &entry.metadata.checksum {
                    let current_checksum = self.calculate_checksum(&entry.data);
                    Ok(stored_checksum == &current_checksum)
                } else {
                    Ok(true)
                }
            }
            CacheValidation::StrongConsistency => {
                // Implement strong consistency validation
                Ok(true)
            }
        }
    }

    fn calculate_checksum(&self, data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    fn get_memory_usage(&self) -> usize {
        self.memory_store.iter()
            .map(|entry| entry.metadata.size)
            .sum()
    }
}

impl CacheMetrics {
    fn new() -> Self {
        Self {
            hits: counter!("cache_hits_total"),
            misses: counter!("cache_misses_total"),
            evictions: counter!("cache_evictions_total"),
            memory_usage: gauge!("cache_memory_usage_bytes"),
            disk_usage: gauge!("cache_disk_usage_bytes"),
            compression_ratio: histogram!("cache_compression_ratio"),
            operation_latency: histogram!("cache_operation_latency_seconds"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::test;
    use std::time::Duration;
    use pretty_assertions::assert_eq;

    #[test]
    async fn test_cache_operations() {
        let config = CacheConfig {
            max_memory_size: 1024 * 1024,
            max_disk_size: 1024 * 1024 * 10,
            eviction_policy: EvictionPolicy::LRU,
            compression_level: CompressionLevel::Default,
            persistence_path: None,
            cache_validation: CacheValidation::Checksum,
            ttl: None,
        };

        let cache = Cache::new(config).await.unwrap();

        // Test basic set/get operations
        cache.set("key1".to_string(), "value1", None).await.unwrap();
        let result: String = cache.get("key1").await.unwrap().unwrap();
        assert_eq!(result, "value1");

        // Test non-existent key
        let result: Option<String> = cache.get("nonexistent").await.unwrap();
        assert!(result.is_none());

        // Test removal
        cache.remove("key1").await.unwrap();
        let result: Option<String> = cache.get("key1").await.unwrap();
        assert!(result.is_none());
    }

    #[test]
    async fn test_cache_eviction() {
        let config = CacheConfig {
            max_memory_size: 100, // Small size to force eviction
            max_disk_size: 1000,
            eviction_policy: EvictionPolicy::LRU,
            compression_level: CompressionLevel::None,
            persistence_path: None,
            cache_validation: CacheValidation::None,
            ttl: None,
        };

        let cache = Cache::new(config).await.unwrap();

        // Fill cache beyond capacity
        for i in 0..10 {
            cache.set(
                format!("key{}", i),
                vec![0u8; 20], // Each entry is 20 bytes
                None
            ).await.unwrap();
        }

        // Verify older entries were evicted
        let result: Option<Vec<u8>> = cache.get("key0").await.unwrap();
        assert!(result.is_none());

        // Verify newer entries are still present
        let result: Option<Vec<u8>> = cache.get("key9").await.unwrap();
        assert!(result.is_some());
    }

    #[test]
    async fn test_cache_ttl() {
        let config = CacheConfig {
            max_memory_size: 1024 * 1024,
            max_disk_size: 1024 * 1024,
            eviction_policy: EvictionPolicy::LRU,
            compression_level: CompressionLevel::None,
            persistence_path: None,
            cache_validation: CacheValidation::None,
            ttl: Some(Duration::from_millis(100)),
        };

        let cache = Cache::new(config).await.unwrap();

        // Set value with TTL
        cache.set(
            "ttl_key".to_string(),
            "ttl_value",
            Some(Duration::from_millis(50))
        ).await.unwrap();

        // Verify value exists initially
        let result: Option<String> = cache.get("ttl_key").await.unwrap();
        assert!(result.is_some());

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify value has expired
        let result: Option<String> = cache.get("ttl_key").await.unwrap();
        assert!(result.is_none());
    }

    #[test]
    async fn test_cache_persistence() {
        let temp_dir = tempdir().unwrap();
        let config = CacheConfig {
            max_memory_size: 1024 * 1024,
            max_disk_size: 1024 * 1024,
            eviction_policy: EvictionPolicy::LRU,
            compression_level: CompressionLevel::Default,
            persistence_path: Some(temp_dir.path().to_path_buf()),
            cache_validation: CacheValidation::Checksum,
            ttl: None,
        };

        // Create cache and store value
        let cache = Cache::new(config.clone()).await.unwrap();
        cache.set("persist_key".to_string(), "persist_value", None).await.unwrap();

        // Create new cache instance with same config
        let cache2 = Cache::new(config).await.unwrap();
        let result: String = cache2.get("persist_key").await.unwrap().unwrap();
        assert_eq!(result, "persist_value");
    }

    #[test]
    async fn test_cache_compression() {
        let config = CacheConfig {
            max_memory_size: 1024 * 1024,
            max_disk_size: 1024 * 1024,
            eviction_policy: EvictionPolicy::LRU,
            compression_level: CompressionLevel::Best,
            persistence_path: None,
            cache_validation: CacheValidation::None,
            ttl: None,
        };

        let cache = Cache::new(config).await.unwrap();

        // Create compressible data
        let data = vec![0u8; 1000];
        cache.set("compress_key".to_string(), data.clone(), None).await.unwrap();

        // Verify data can be retrieved correctly
        let result: Vec<u8> = cache.get("compress_key").await.unwrap().unwrap();
        assert_eq!(result, data);

        // Verify compression occurred
        if let Some(entry) = cache.memory_store.get("compress_key") {
            assert!(entry.compression_info.is_some());
            let compression_info = entry.compression_info.as_ref().unwrap();
            assert!(compression_info.compressed_size < compression_info.original_size);
        }
    }

    #[test]
    async fn test_cache_metrics() {
        let config = CacheConfig::default();
        let cache = Cache::new(config).await.unwrap();

        // Generate some cache operations
        cache.set("metrics_key".to_string(), "value", None).await.unwrap();
        let _: Option<String> = cache.get("metrics_key").await.unwrap();
        let _: Option<String> = cache.get("nonexistent").await.unwrap();

        // Verify metrics were recorded
        assert!(cache.metrics.hits.get() > 0);
        assert!(cache.metrics.misses.get() > 0);
        assert!(cache.metrics.memory_usage.get() > 0.0);
    }

    #[test]
    async fn test_concurrent_access() {
        let config = CacheConfig::default();
        let cache = Arc::new(Cache::new(config).await.unwrap());

        let mut handles = Vec::new();
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                cache_clone.set(
                    format!("concurrent_key_{}", i),
                    format!("value_{}", i),
                    None
                ).await.unwrap();

                let result: String = cache_clone
                    .get(&format!("concurrent_key_{}", i))
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(result, format!("value_{}", i));
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }
    }

    #[test]
    async fn test_eviction_policies() {
        async fn test_policy(policy: EvictionPolicy) {
            let config = CacheConfig {
                max_memory_size: 100,
                eviction_policy: policy,
                ..CacheConfig::default()
            };

            let cache = Cache::new(config).await.unwrap();

            // Fill cache and trigger eviction
            for i in 0..10 {
                cache.set(
                    format!("policy_key_{}", i),
                    vec![0u8; 20],
                    None
                ).await.unwrap();
                
                // Access some keys more frequently for LFU testing
                if i % 2 == 0 {
                    let _: Option<Vec<u8>> = cache.get(&format!("policy_key_{}", i)).await.unwrap();
                }
            }

            // Verify eviction behavior matches policy
            match policy {
                EvictionPolicy::LRU => {
                    // Least recently used should be evicted
                    let result: Option<Vec<u8>> = cache.get("policy_key_0").await.unwrap();
                    assert!(result.is_none());
                }
                EvictionPolicy::LFU => {
                    // Least frequently used should be evicted
                    let result: Option<Vec<u8>> = cache.get("policy_key_1").await.unwrap();
                    assert!(result.is_none());
                }
                _ => {}
            }
        }

        test_policy(EvictionPolicy::LRU).await;
        test_policy(EvictionPolicy::LFU).await;
    }

    proptest::proptest! {
        #[test]
        fn test_cache_with_random_data(
            key in "\\PC{1,100}",
            value in proptest::collection::vec(any::<u8>(), 0..1000)
        ) {
            let config = CacheConfig::default();
            let cache = tokio_test::block_on(Cache::new(config)).unwrap();
            
            tokio_test::block_on(async {
                cache.set(key.clone(), value.clone(), None).await.unwrap();
                let result: Vec<u8> = cache.get(&key).await.unwrap().unwrap();
                prop_assert_eq!(result, value);
            });
        }
    }
}