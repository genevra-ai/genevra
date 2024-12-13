use std::{
    path::{Path, PathBuf},
    sync::Arc,
    collections::{HashMap, BTreeMap},
    str::FromStr,
    time::Duration,
};

use anyhow::{Result, Context, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use config::{Config as ConfigBuilder, Environment, File, FileFormat};
use tracing::{debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub hardware: HardwareConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub network: NetworkConfig,
    #[serde(default)]
    pub security: SecurityConfig,
    #[serde(default)]
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub max_threads: usize,
    pub task_queue_size: usize,
    pub scheduling_policy: SchedulingPolicy,
    pub timeout_ms: u64,
    pub backoff_strategy: BackoffStrategy,
    #[serde(with = "humantime_serde")]
    pub grace_period: Duration,
    pub panic_strategy: PanicStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub use_gpu: bool,
    pub gpu_memory_limit: Option<usize>,
    pub cpu_affinity: Vec<usize>,
    pub numa_policy: NumaPolicy,
    pub vectorization: VectorizationPolicy,
    pub memory_limit: ByteSize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub temp_dir: PathBuf,
    pub compression_level: CompressionLevel,
    pub io_engine: IoEngine,
    pub buffer_size: ByteSize,
    pub fsync_policy: FSyncPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub bind_address: String,
    pub port_range: (u16, u16),
    pub keepalive_interval: Duration,
    pub tcp_nodelay: bool,
    pub backlog: u32,
    pub tls: Option<TlsConfig>,
    pub proxy_protocol: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub encryption_key: SecretString,
    pub auth_methods: Vec<AuthMethod>,
    pub rate_limits: RateLimitConfig,
    pub ip_whitelist: Vec<String>,
    pub audit_log: bool,
    pub sandbox_config: SandboxConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_port: u16,
    pub tracing_level: TracingLevel,
    pub sampling_rate: f64,
    pub exporters: Vec<MetricsExporter>,
    pub health_check_interval: Duration,
    pub alerting: AlertingConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SchedulingPolicy {
    Fifo,
    RoundRobin,
    WeightedRoundRobin(u32),
    Priority,
    Custom(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackoffStrategy {
    None,
    Linear { initial_ms: u64, step_ms: u64, max_attempts: u32 },
    Exponential { initial_ms: u64, multiplier: f64, max_ms: u64 },
    Fibonacci { initial_ms: u64, max_attempts: u32 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PanicStrategy {
    Abort,
    Unwind,
    Restart { max_attempts: u32 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NumaPolicy {
    Local,
    Interleave,
    Preferred(u32),
    Strict,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VectorizationPolicy {
    Auto,
    Force,
    Disable,
    Runtime,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionLevel {
    None,
    Fast,
    Default,
    Best,
    Custom(u32),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IoEngine {
    Sync,
    AsyncIo,
    Uring,
    MMap,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FSyncPolicy {
    Never,
    Always,
    OnClose,
    Periodic(Duration),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
    pub verify_client: bool,
    pub ciphers: Vec<String>,
    pub min_version: TlsVersion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_second: u32,
    pub burst_size: u32,
    pub per_ip: bool,
    pub exempt_ips: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub enable: bool,
    pub memory_limit: ByteSize,
    pub cpu_quota: f64,
    pub allowed_syscalls: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TracingLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsExporter {
    Prometheus { port: u16 },
    StatsD { host: String, port: u16 },
    OpenTelemetry { endpoint: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub endpoints: Vec<AlertEndpoint>,
    pub rules: Vec<AlertRule>,
    pub aggregation_window: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEndpoint {
    pub name: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub condition: String,
    pub threshold: f64,
    pub window: Duration,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteSize(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretString(String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    None,
    Basic { realm: String },
    Bearer { jwt_secret: SecretString },
    Custom { name: String, config: BTreeMap<String, String> },
}

pub struct ConfigManager {
    config: Arc<RwLock<SystemConfig>>,
    file_path: PathBuf,
    environment: String,
    validators: Vec<Box<dyn ConfigValidator + Send + Sync>>,
}

#[async_trait::async_trait]
pub trait ConfigValidator: std::fmt::Debug {
    async fn validate(&self, config: &SystemConfig) -> Result<()>;
}

impl ConfigManager {
    pub async fn new<P: AsRef<Path>>(
        file_path: P,
        environment: &str,
    ) -> Result<Self> {
        let file_path = file_path.as_ref().to_owned();
        let config = Self::load_config(&file_path, environment).await?;
        
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            file_path,
            environment: environment.to_owned(),
            validators: Vec::new(),
        })
    }

    pub async fn reload(&self) -> Result<()> {
        let new_config = Self::load_config(&self.file_path, &self.environment).await?;
        
        for validator in &self.validators {
            validator.validate(&new_config).await?;
        }
        
        let mut config = self.config.write().await;
        *config = new_config;
        
        Ok(())
    }

    async fn load_config(path: &Path, environment: &str) -> Result<SystemConfig> {
        let builder = ConfigBuilder::builder()
            .add_source(File::from(path).required(true))
            .add_source(Environment::with_prefix("APP").separator("__"))
            .add_source(File::new(
                &format!("config.{}.toml", environment),
                FileFormat::Toml,
            ).required(false))
            .build()?;

        let config: SystemConfig = builder.try_deserialize()?;
        debug!("Loaded configuration for environment: {}", environment);
        
        Ok(config)
    }

    pub fn add_validator<V: ConfigValidator + Send + Sync + 'static>(&mut self, validator: V) {
        self.validators.push(Box::new(validator));
    }

    pub async fn get_config(&self) -> Arc<SystemConfig> {
        Arc::new(self.config.read().await.clone())
    }

    pub async fn update_config<F, T>(&self, update_fn: F) -> Result<T>
    where
        F: FnOnce(&mut SystemConfig) -> Result<T>,
    {
        let mut config = self.config.write().await;
        let result = update_fn(&mut config)?;
        
        for validator in &self.validators {
            validator.validate(&config).await?;
        }
        
        Ok(result)
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_threads: num_cpus::get(),
            task_queue_size: 10_000,
            scheduling_policy: SchedulingPolicy::RoundRobin,
            timeout_ms: 30_000,
            backoff_strategy: BackoffStrategy::Exponential {
                initial_ms: 100,
                multiplier: 2.0,
                max_ms: 30_000,
            },
            grace_period: Duration::from_secs(30),
            panic_strategy: PanicStrategy::Unwind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::fs;

    #[tokio::test]
    async fn test_config_loading() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        
        let config_content = r#"
            [runtime]
            max_threads = 4
            task_queue_size = 1000
            timeout_ms = 5000
            
            [hardware]
            use_gpu = true
            cpu_affinity = [0, 1]
            
            [storage]
            data_dir = "/data"
            cache_dir = "/cache"
            temp_dir = "/tmp"
        "#;
        
        fs::write(&config_path, config_content).await.unwrap();
        
        let manager = ConfigManager::new(&config_path, "test").await.unwrap();
        let config = manager.get_config().await;
        
        assert_eq!(config.runtime.max_threads, 4);
        assert_eq!(config.hardware.use_gpu, true);
        assert_eq!(config.hardware.cpu_affinity, vec![0, 1]);
    }

    #[tokio::test]
    async fn test_config_validation() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        
        let config_content = r#"
            [runtime]
            max_threads = 4
        "#;
        
        fs::write(&config_path, config_content).await.unwrap();
        
        let mut manager = ConfigManager::new(&config_path, "test").await.unwrap();
        
        #[derive(Debug)]
        struct TestValidator;
        
        #[async_trait::async_trait]
        impl ConfigValidator for TestValidator {
            async fn validate(&self, config: &SystemConfig) -> Result<()> {
                if config.runtime.max_threads < 1 {
                    bail!("max_threads must be at least 1");
                }
                Ok(())
            }
        }
        
        manager.add_validator(TestValidator);
        
        assert!(manager.reload().await.is_ok());
    }
}