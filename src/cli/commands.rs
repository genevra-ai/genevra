use std::{
    path::{Path, PathBuf},
    sync::Arc,
    collections::HashMap,
    time::Duration,
};

use anyhow::{Result, Context, bail};
use tokio::{fs, sync::mpsc};
use futures::{Stream, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use tracing::{info, warn, error, debug};
use serde::{Serialize, Deserialize};

use crate::{
    genomics::{
        SequenceAnalyzer, SequenceFormat, AnalysisConfig, AnalysisResult,
        VariantCaller, ReferenceGenome,
    },
    ai::{
        ModelConfig, ModelRegistry, TrainingConfig, InferenceConfig,
        TrainingPipeline, InferencePipeline,
    },
    utils::{self, io::FileReader},
    config::GlobalConfig,
};

#[derive(Debug, Deserialize)]
pub struct BatchConfig {
    pub max_concurrent: usize,
    pub timeout: Duration,
    pub retry_count: usize,
    pub validation: ValidationConfig,
}

#[derive(Debug, Deserialize)]
pub struct ValidationConfig {
    pub min_quality: f64,
    pub min_coverage: u32,
    pub max_error_rate: f64,
}

#[derive(Debug, Serialize)]
pub struct CommandResult {
    pub status: String,
    pub duration: Duration,
    pub details: HashMap<String, String>,
}

pub async fn handle_analyze_command(
    input_files: Vec<PathBuf>,
    output_dir: PathBuf,
    config: AnalysisConfig,
    format: SequenceFormat,
) -> Result<CommandResult> {
    let start_time = std::time::Instant::now();
    let analyzer = Arc::new(SequenceAnalyzer::new(config.clone())?);
    
    // Create output directory
    fs::create_dir_all(&output_dir).await?;

    // Setup progress bar
    let pb = ProgressBar::new(input_files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap()
        .progress_chars("##-"));

    // Process files concurrently
    let (tx, mut rx) = mpsc::channel(config.max_concurrent_jobs);
    
    for input_file in input_files {
        let analyzer = Arc::clone(&analyzer);
        let tx = tx.clone();
        let output_dir = output_dir.clone();
        
        tokio::spawn(async move {
            match process_file(input_file, &output_dir, &analyzer, format).await {
                Ok(result) => tx.send(Ok(result)).await,
                Err(e) => tx.send(Err(e)).await,
            }
        });
    }
    drop(tx);

    let mut results = Vec::new();
    while let Some(result) = rx.recv().await {
        match result {
            Ok(analysis_result) => {
                results.push(analysis_result);
                pb.inc(1);
            }
            Err(e) => {
                error!("Analysis error: {}", e);
                pb.println(format!("Error: {}", e));
            }
        }
    }

    pb.finish_with_message("Analysis complete");

    // Generate summary report
    let summary = generate_analysis_summary(&results)?;
    let summary_path = output_dir.join("analysis_summary.json");
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?).await?;

    Ok(CommandResult {
        status: "completed".to_string(),
        duration: start_time.elapsed(),
        details: HashMap::from([
            ("files_processed".to_string(), results.len().to_string()),
            ("summary_path".to_string(), summary_path.display().to_string()),
        ]),
    })
}

pub async fn handle_train_command(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_dir: PathBuf,
    output_dir: PathBuf,
) -> Result<CommandResult> {
    let start_time = std::time::Instant::now();
    
    // Initialize training pipeline
    let pipeline = TrainingPipeline::new(model_config.clone(), training_config.clone())?;
    
    // Setup progress tracking
    let pb = ProgressBar::new(training_config.epochs as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.green/blue} Epoch {pos:>3}/{len:3} {msg}")
        .unwrap());

    // Training loop with progress updates
    let (tx, mut rx) = mpsc::channel(1);
    let training_handle = tokio::spawn(async move {
        pipeline.train(data_dir, output_dir, Some(tx)).await
    });

    while let Some(progress) = rx.recv().await {
        pb.set_position(progress.epoch as u64);
        pb.set_message(format!(
            "Loss: {:.4} | Accuracy: {:.2}%",
            progress.loss,
            progress.accuracy * 100.0
        ));
    }

    let training_result = training_handle.await??;
    pb.finish_with_message("Training complete");

    Ok(CommandResult {
        status: "completed".to_string(),
        duration: start_time.elapsed(),
        details: HashMap::from([
            ("final_loss".to_string(), format!("{:.4}", training_result.final_loss)),
            ("final_accuracy".to_string(), format!("{:.2}%", training_result.final_accuracy * 100.0)),
            ("model_path".to_string(), training_result.model_path.display().to_string()),
        ]),
    })
}

pub async fn handle_inference_command(
    model_id: String,
    input_files: Vec<PathBuf>,
    output_dir: PathBuf,
    config: InferenceConfig,
) -> Result<CommandResult> {
    let start_time = std::time::Instant::now();
    
    // Load model
    let registry = ModelRegistry::new().await?;
    let model = registry.load_model(&model_id).await?;
    
    // Initialize inference pipeline
    let pipeline = InferencePipeline::new(model, config.clone())?;
    
    // Setup progress tracking
    let pb = ProgressBar::new(input_files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.yellow/blue} {pos:>7}/{len:7} {msg}")
        .unwrap());

    // Process files in batches
    let mut results = Vec::new();
    for batch in input_files.chunks(config.batch_size) {
        let batch_results = pipeline.process_batch(batch).await?;
        results.extend(batch_results);
        pb.inc(batch.len() as u64);
    }

    pb.finish_with_message("Inference complete");

    // Save results
    let results_path = output_dir.join("inference_results.json");
    fs::write(&results_path, serde_json::to_string_pretty(&results)?).await?;

    Ok(CommandResult {
        status: "completed".to_string(),
        duration: start_time.elapsed(),
        details: HashMap::from([
            ("files_processed".to_string(), results.len().to_string()),
            ("results_path".to_string(), results_path.display().to_string()),
        ]),
    })
}

pub async fn handle_model_list_command(format: Option<String>) -> Result<CommandResult> {
    let registry = ModelRegistry::new().await?;
    let models = registry.list_models().await?;

    match format.as_deref() {
        Some("json") => {
            println!("{}", serde_json::to_string_pretty(&models)?);
        }
        Some("table") => {
            print_model_table(&models);
        }
        _ => {
            for model in &models {
                println!("{} (v{}) - {}", model.name, model.version, model.description);
            }
        }
    }

    Ok(CommandResult {
        status: "completed".to_string(),
        duration: Duration::default(),
        details: HashMap::from([
            ("model_count".to_string(), models.len().to_string()),
        ]),
    })
}

pub async fn handle_model_download_command(
    model_id: String,
    version: Option<String>,
) -> Result<CommandResult> {
    let start_time = std::time::Instant::now();
    let registry = ModelRegistry::new().await?;

    let pb = ProgressBar::new_spinner();
    pb.set_message(format!("Downloading model {}", model_id));
    pb.enable_steady_tick(Duration::from_millis(100));

    let model_info = registry.download_model(&model_id, version.as_deref()).await?;
    pb.finish_with_message("Download complete");

    Ok(CommandResult {
        status: "completed".to_string(),
        duration: start_time.elapsed(),
        details: HashMap::from([
            ("model_path".to_string(), model_info.path.display().to_string()),
            ("model_size".to_string(), format!("{} bytes", model_info.size)),
        ]),
    })
}

async fn process_file(
    input_file: PathBuf,
    output_dir: &Path,
    analyzer: &SequenceAnalyzer,
    format: SequenceFormat,
) -> Result<AnalysisResult> {
    let reader = FileReader::open(&input_file).await?;
    let sequence = reader.read_sequence(format).await?;
    
    let result = analyzer.analyze_sequence(&sequence).await?;
    
    let output_file = output_dir.join(input_file.file_name().unwrap())
        .with_extension("analysis.json");
    
    fs::write(&output_file, serde_json::to_string_pretty(&result)?).await?;
    
    Ok(result)
}

fn generate_analysis_summary(results: &[AnalysisResult]) -> Result<serde_json::Value> {
    let summary = serde_json::json!({
        "total_sequences": results.len(),
        "average_gc_content": results.iter()
            .map(|r| r.gc_content)
            .sum::<f64>() / results.len() as f64,
        "average_quality": results.iter()
            .filter_map(|r| r.average_quality)
            .sum::<f64>() / results.len() as f64,
        "variant_statistics": {
            "total_variants": results.iter()
                .map(|r| r.variants.len())
                .sum::<usize>(),
            "snp_count": results.iter()
                .flat_map(|r| &r.variants)
                .filter(|v| v.is_snp())
                .count(),
            "indel_count": results.iter()
                .flat_map(|r| &r.variants)
                .filter(|v| v.is_indel())
                .count(),
        },
    });

    Ok(summary)
}

fn print_model_table(models: &[ModelInfo]) {
    use prettytable::{Table, Row, Cell};

    let mut table = Table::new();
    table.add_row(Row::new(vec![
        Cell::new("ID"),
        Cell::new("Name"),
        Cell::new("Version"),
        Cell::new("Status"),
        Cell::new("Last Updated"),
    ]));

    for model in models {
        table.add_row(Row::new(vec![
            Cell::new(&model.id),
            Cell::new(&model.name),
            Cell::new(&model.version.to_string()),
            Cell::new(&model.status.to_string()),
            Cell::new(&model.updated_at.to_rfc3339()),
        ]));
    }

    table.printstd();
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_fs::prelude::*;
    use predicates::prelude::*;

    #[tokio::test]
    async fn test_analyze_command() {
        let temp = assert_fs::TempDir::new().unwrap();
        let input = temp.child("test.fa");
        input.write_str(">test\nATCG\n").unwrap();
        
        let output_dir = temp.child("output");
        let config = AnalysisConfig::default();

        let result = handle_analyze_command(
            vec![input.path().to_owned()],
            output_dir.path().to_owned(),
            config,
            SequenceFormat::Fasta,
        ).await.unwrap();

        assert_eq!(result.status, "completed");
        output_dir.child("analysis_summary.json").assert(predicate::path::exists());
    }

    #[tokio::test]
    async fn test_model_list_command() {
        let result = handle_model_list_command(Some("json".to_string())).await.unwrap();
        assert_eq!(result.status, "completed");
    }

    #[test]
    fn test_generate_analysis_summary() {
        let results = vec![
            AnalysisResult {
                gc_content: 0.5,
                average_quality: Some(30.0),
                variants: vec![],
                ..Default::default()
            },
            AnalysisResult {
                gc_content: 0.6,
                average_quality: Some(35.0),
                variants: vec![],
                ..Default::default()
            },
        ];

        let summary = generate_analysis_summary(&results).unwrap();
        assert_eq!(summary["total_sequences"], 2);
        assert!(summary["average_gc_content"].as_f64().unwrap() > 0.0);
    }
}