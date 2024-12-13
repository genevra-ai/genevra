use std::{
    sync::Arc,
    collections::{HashMap, BTreeMap},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Result, Context, bail};
use tokio::{sync::RwLock, fs};
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use semver::Version;

use crate::models::{Model, ModelConfig, ModelArchitecture};
use super::metrics::ModelMetrics;

const MAX_MODEL_VERSIONS: usize = 5;
const REGISTRY_FILE: &str = "model_registry.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub version: Version,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub architecture: ModelArchitecture,
    pub config: ModelConfig,
    pub checksum: String,
    pub size_bytes: u64,
    pub metrics: Option<ModelMetrics>,
    pub tags: Vec<String>,
    pub status: ModelStatus,
    pub dependencies: Vec<ModelDependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDependency {
    pub model_id: String,
    pub version_req: semver::VersionReq,
    pub optional: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    Active,
    Deprecated,
    Archived,
    Testing,
    Failed,
}

pub struct ModelRegistry {
    base_path: PathBuf,
    models: Arc<RwLock<BTreeMap<String, HashMap<Version, ModelMetadata>>>>,
    loaded_models: Arc<RwLock<HashMap<String, Arc<dyn Model>>>>,
    cache_config: RegistryCacheConfig,
}

#[derive(Debug, Clone)]
struct RegistryCacheConfig {
    max_cached_models: usize,
    ttl: Duration,
    eviction_policy: CacheEvictionPolicy,
}

#[derive(Debug, Clone, Copy)]
enum CacheEvictionPolicy {
    LRU,
    LFU,
    Random,
}

impl ModelRegistry {
    pub async fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref().to_owned();
        fs::create_dir_all(&base_path).await?;

        let registry_path = base_path.join(REGISTRY_FILE);
        let models = if registry_path.exists() {
            let content = fs::read_to_string(&registry_path).await?;
            serde_json::from_str(&content)?
        } else {
            BTreeMap::new()
        };

        Ok(Self {
            base_path,
            models: Arc::new(RwLock::new(models)),
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            cache_config: RegistryCacheConfig {
                max_cached_models: 10,
                ttl: Duration::from_secs(3600),
                eviction_policy: CacheEvictionPolicy::LRU,
            },
        })
    }

    pub async fn register_model(
        &self,
        name: String,
        version: Version,
        model: Arc<dyn Model>,
        config: ModelConfig,
        tags: Vec<String>,
    ) -> Result<ModelMetadata> {
        let model_path = self.get_model_path(&name, &version);
        let checksum = self.compute_model_checksum(&model).await?;
        
        let metadata = ModelMetadata {
            id: format!("{}-{}", name, version),
            name: name.clone(),
            version: version.clone(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            architecture: model.architecture(),
            config,
            checksum,
            size_bytes: self.get_model_size(&model)?,
            metrics: None,
            tags,
            status: ModelStatus::Testing,
            dependencies: Vec::new(),
        };

        // Save model
        model.save(&model_path).await?;

        // Update registry
        let mut models = self.models.write().await;
        models
            .entry(name.clone())
            .or_default()
            .insert(version.clone(), metadata.clone());

        // Persist registry
        self.save_registry().await?;

        Ok(metadata)
    }

    pub async fn load_model(&self, name: &str, version: Option<&Version>) -> Result<Arc<dyn Model>> {
        let models = self.models.read().await;
        let model_versions = models.get(name).context("Model not found")?;

        let version = if let Some(v) = version {
            v.clone()
        } else {
            model_versions
                .keys()
                .max()
                .context("No versions available")?
                .clone()
        };

        let metadata = model_versions.get(&version).context("Version not found")?;
        
        // Check cache
        if let Some(model) = self.loaded_models.read().await.get(&metadata.id) {
            return Ok(model.clone());
        }

        // Load model
        let model_path = self.get_model_path(&name, &version);
        let mut model = self.create_model(&metadata.architecture, &metadata.config)?;
        model.load(&model_path).await?;

        // Verify checksum
        let checksum = self.compute_model_checksum(&model).await?;
        if checksum != metadata.checksum {
            bail!("Model checksum mismatch");
        }

        let model = Arc::new(model);

        // Update cache
        let mut loaded_models = self.loaded_models.write().await;
        self.manage_cache(&mut loaded_models).await?;
        loaded_models.insert(metadata.id.clone(), model.clone());

        Ok(model)
    }

    pub async fn update_metrics(&self, name: &str, version: &Version, metrics: &ModelMetrics) -> Result<()> {
        let mut models = self.models.write().await;
        let model_versions = models.get_mut(name).context("Model not found")?;
        let metadata = model_versions.get_mut(version).context("Version not found")?;

        metadata.metrics = Some(metrics.clone());
        metadata.updated_at = chrono::Utc::now();

        self.save_registry().await?;
        Ok(())
    }

    pub async fn update_status(&self, name: &str, version: &Version, status: ModelStatus) -> Result<()> {
        let mut models = self.models.write().await;
        let model_versions = models.get_mut(name).context("Model not found")?;
        let metadata = model_versions.get_mut(version).context("Version not found")?;

        metadata.status = status;
        metadata.updated_at = chrono::Utc::now();

        self.save_registry().await?;
        Ok(())
    }

    pub async fn add_dependency(
        &self,
        name: &str,
        version: &Version,
        dependency: ModelDependency,
    ) -> Result<()> {
        let mut models = self.models.write().await;
        let model_versions = models.get_mut(name).context("Model not found")?;
        let metadata = model_versions.get_mut(version).context("Version not found")?;

        // Verify dependency exists
        let dep_versions = models.get(&dependency.model_id).context("Dependency not found")?;
        let compatible_version = dep_versions
            .keys()
            .find(|v| dependency.version_req.matches(v))
            .context("No compatible version found for dependency")?;

        metadata.dependencies.push(dependency);
        metadata.updated_at = chrono::Utc::now();

        self.save_registry().await?;
        Ok(())
    }

    pub async fn get_metadata(&self, name: &str, version: Option<&Version>) -> Result<ModelMetadata> {
        let models = self.models.read().await;
        let model_versions = models.get(name).context("Model not found")?;

        let version = if let Some(v) = version {
            v.clone()
        } else {
            model_versions
                .keys()
                .max()
                .context("No versions available")?
                .clone()
        };

        model_versions
            .get(&version)
            .cloned()
            .context("Version not found")
    }

    pub async fn list_models(&self) -> Result<Vec<ModelMetadata>> {
        let models = self.models.read().await;
        let mut result = Vec::new();

        for versions in models.values() {
            for metadata in versions.values() {
                result.push(metadata.clone());
            }
        }

        Ok(result)
    }

    pub async fn search_models(&self, query: &str, tags: Option<&[String]>) -> Result<Vec<ModelMetadata>> {
        let models = self.models.read().await;
        let mut result = Vec::new();

        for versions in models.values() {
            for metadata in versions.values() {
                if metadata.name.contains(query) || metadata.tags.iter().any(|t| t.contains(query)) {
                    if let Some(required_tags) = tags {
                        if required_tags.iter().all(|t| metadata.tags.contains(t)) {
                            result.push(metadata.clone());
                        }
                    } else {
                        result.push(metadata.clone());
                    }
                }
            }
        }

        Ok(result)
    }

    async fn save_registry(&self) -> Result<()> {
        let registry_path = self.base_path.join(REGISTRY_FILE);
        let models = self.models.read().await;
        let content = serde_json::to_string_pretty(&*models)?;
        fs::write(registry_path, content).await?;
        Ok(())
    }

    fn get_model_path(&self, name: &str, version: &Version) -> PathBuf {
        self.base_path.join(format!("{}_{}.pt", name, version))
    }

    async fn compute_model_checksum(&self, model: &Arc<dyn Model>) -> Result<String> {
        let mut hasher = Sha256::new();
        for param in model.parameters() {
            hasher.update(&*param.data());
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn get_model_size(&self, model: &Arc<dyn Model>) -> Result<u64> {
        let mut size = 0;
        for param in model.parameters() {
            size += param.data().len() as u64;
        }
        Ok(size)
    }

    async fn manage_cache(&self, loaded_models: &mut HashMap<String, Arc<dyn Model>>) -> Result<()> {
        if loaded_models.len() >= self.cache_config.max_cached_models {
            match self.cache_config.eviction_policy {
                CacheEvictionPolicy::LRU => {
                    // Implement LRU eviction
                    if let Some(oldest) = loaded_models.keys().next().cloned() {
                        loaded_models.remove(&oldest);
                    }
                }
                CacheEvictionPolicy::LFU => {
                    // Implement LFU eviction
                    // For simplicity, just remove first entry
                    if let Some(first) = loaded_models.keys().next().cloned() {
                        loaded_models.remove(&first);
                    }
                }
                CacheEvictionPolicy::Random => {
                    // Remove random entry
                    if let Some(random_key) = loaded_models.keys().next().cloned() {
                        loaded_models.remove(&random_key);
                    }
                }
            }
        }
        Ok(())
    }

    fn create_model(&self, architecture: &ModelArchitecture, config: &ModelConfig) -> Result<Box<dyn Model>> {
        // Factory pattern for model creation based on architecture
        match architecture {
            ModelArchitecture::Transformer(cfg) => {
                // Create transformer model
                unimplemented!()
            }
            ModelArchitecture::CNN(cfg) => {
                // Create CNN model
                unimplemented!()
            }
            _ => bail!("Unsupported model architecture"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_model_registration() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).await.unwrap();

        let model = Arc::new(DummyModel::new());
        let config = ModelConfig::default();
        let version = Version::new(1, 0, 0);

        let metadata = registry
            .register_model(
                "test_model".to_string(),
                version.clone(),
                model,
                config,
                vec!["test".to_string()],
            )
            .await
            .unwrap();

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.version, version);
        assert_eq!(metadata.status, ModelStatus::Testing);
    }

    #[tokio::test]
    async fn test_model_loading() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).await.unwrap();

        let model = Arc::new(DummyModel::new());
        let config = ModelConfig::default();
        let version = Version::new(1, 0, 0);

        registry
            .register_model(
                "test_model".to_string(),
                version.clone(),
                model,
                config,
                vec![],
            )
            .await
            .unwrap();

        let loaded_model = registry.load_model("test_model", Some(&version)).await.unwrap();
        assert!(loaded_model.architecture() == ModelArchitecture::Custom("Dummy".to_string()));
    }

    #[tokio::test]
    async fn test_model_versioning() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).await.unwrap();

        let model = Arc::new(DummyModel::new());
        let config = ModelConfig::default();

        // Register multiple versions
        let versions = vec![
            Version::new(1, 0, 0),
            Version::new(1, 1, 0),
            Version::new(2, 0, 0),
        ];

        for version in &versions {
            registry
                .register_model(
                    "test_model".to_string(),
                    version.clone(),
                    model.clone(),
                    config.clone(),
                    vec![],
                )
                .await
                .unwrap();
        }

        // Test latest version loading
        let latest = registry.load_model("test_model", None).await.unwrap();
        let metadata = registry.get_metadata("test_model", None).await.unwrap();
        assert_eq!(metadata.version, versions.last().unwrap());
    }

    #[tokio::test]
    async fn test_model_search() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).await.unwrap();

        let model = Arc::new(DummyModel::new());
        let config = ModelConfig::default();

        // Register models with different tags
        registry
            .register_model(
                "transformer_model".to_string(),
                Version::new(1, 0, 0),
                model.clone(),
                config.clone(),
                vec!["nlp".to_string(), "transformer".to_string()],
            )
            .await
            .unwrap();

        registry
            .register_model(
                "cnn_model".to_string(),
                Version::new(1, 0, 0),
                model.clone(),
                config.clone(),
                vec!["vision".to_string(), "cnn".to_string()],
            )
            .await
            .unwrap();

        // Test search by name
        let results = registry.search_models("transformer", None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "transformer_model");

        // Test search by tag
        let results = registry
            .search_models("", Some(&vec!["nlp".to_string()]))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tags, vec!["nlp", "transformer"]);

        // Test search with multiple tags
        let results = registry
            .search_models(
                "",
                Some(&vec!["vision".to_string(), "cnn".to_string()]),
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "cnn_model");

        // Test search with no matches
        let results = registry
            .search_models("nonexistent", None)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_model_status_update() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).await.unwrap();

        let model = Arc::new(DummyModel::new());
        let config = ModelConfig::default();
        let version = Version::new(1, 0, 0);

        registry
            .register_model(
                "test_model".to_string(),
                version.clone(),
                model,
                config,
                vec![],
            )
            .await
            .unwrap();

        // Update status
        registry
            .update_status("test_model", &version, ModelStatus::Active)
            .await
            .unwrap();

        let metadata = registry
            .get_metadata("test_model", Some(&version))
            .await
            .unwrap();
        assert_eq!(metadata.status, ModelStatus::Active);
    }

    #[tokio::test]
    async fn test_model_dependencies() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).await.unwrap();

        let model = Arc::new(DummyModel::new());
        let config = ModelConfig::default();

        // Register base model
        registry
            .register_model(
                "base_model".to_string(),
                Version::new(1, 0, 0),
                model.clone(),
                config.clone(),
                vec![],
            )
            .await
            .unwrap();

        // Register dependent model
        let dependent_version = Version::new(1, 0, 0);
        registry
            .register_model(
                "dependent_model".to_string(),
                dependent_version.clone(),
                model.clone(),
                config.clone(),
                vec![],
            )
            .await
            .unwrap();

        // Add dependency
        let dependency = ModelDependency {
            model_id: "base_model".to_string(),
            version_req: semver::VersionReq::parse("^1.0.0").unwrap(),
            optional: false,
        };

        registry
            .add_dependency("dependent_model", &dependent_version, dependency)
            .await
            .unwrap();

        let metadata = registry
            .get_metadata("dependent_model", Some(&dependent_version))
            .await
            .unwrap();
        assert_eq!(metadata.dependencies.len(), 1);
    }

    #[tokio::test]
    async fn test_model_metrics_update() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).await.unwrap();

        let model = Arc::new(DummyModel::new());
        let config = ModelConfig::default();
        let version = Version::new(1, 0, 0);

        registry
            .register_model(
                "test_model".to_string(),
                version.clone(),
                model,
                config,
                vec![],
            )
            .await
            .unwrap();

        // Update metrics
        let metrics = ModelMetrics {
            accuracy: 0.95,
            loss: 0.05,
            training_time: Duration::from_secs(1000),
            last_updated: chrono::Utc::now(),
        };

        registry
            .update_metrics("test_model", &version, &metrics)
            .await
            .unwrap();

        let metadata = registry
            .get_metadata("test_model", Some(&version))
            .await
            .unwrap();
        assert!(metadata.metrics.is_some());
        assert_eq!(metadata.metrics.unwrap().accuracy, 0.95);
    }

    #[tokio::test]
    async fn test_registry_persistence() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).await.unwrap();

        let model = Arc::new(DummyModel::new());
        let config = ModelConfig::default();
        let version = Version::new(1, 0, 0);

        // Register model
        registry
            .register_model(
                "test_model".to_string(),
                version.clone(),
                model,
                config,
                vec![],
            )
            .await
            .unwrap();

        // Create new registry instance
        let new_registry = ModelRegistry::new(temp_dir.path()).await.unwrap();
        let metadata = new_registry
            .get_metadata("test_model", Some(&version))
            .await
            .unwrap();
        
        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.version, version);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).await.unwrap();

        let model = Arc::new(DummyModel::new());
        let config = ModelConfig::default();

        // Register more models than cache capacity
        for i in 0..15 {
            registry
                .register_model(
                    format!("model_{}", i),
                    Version::new(1, 0, 0),
                    model.clone(),
                    config.clone(),
                    vec![],
                )
                .await
                .unwrap();

            // Load model to add it to cache
            registry
                .load_model(&format!("model_{}", i), None)
                .await
                .unwrap();
        }

        let loaded_models = registry.loaded_models.read().await;
        assert!(loaded_models.len() <= registry.cache_config.max_cached_models);
    }

    // Helper struct for testing
    #[derive(Clone)]
    struct DummyModel {}

    impl DummyModel {
        fn new() -> Self {
            Self {}
        }
    }

    #[async_trait::async_trait]
    impl Model for DummyModel {
        async fn forward(&self, input: &Tensor) -> Result<Tensor> {
            Ok(input.shallow_clone())
        }

        async fn backward(&self, gradient: &Tensor) -> Result<()> {
            Ok(())
        }

        async fn save(&self, path: &Path) -> Result<()> {
            Ok(())
        }

        async fn load(&mut self, path: &Path) -> Result<()> {
            Ok(())
        }

        fn parameters(&self) -> Vec<nn::Parameter> {
            vec![]
        }

        fn architecture(&self) -> ModelArchitecture {
            ModelArchitecture::Custom("Dummy".to_string())
        }
    }
}