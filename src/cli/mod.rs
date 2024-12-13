use std::{
    path::{Path, PathBuf},
    str::FromStr,
    time::Duration,
};

use anyhow::{Result, Context, bail};
use clap::{Parser, Subcommand, Args};
use tokio::fs;
use tracing::{info, warn, error};

use crate::{
    genomics::{SequenceFormat, AnalysisConfig},
    ai::{ModelConfig, TrainingConfig, InferenceConfig},
    config::GlobalConfig,
};

#[derive(Parser)]
#[command(name = "genome-ai", version, author, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    #[arg(short, long)]
    pub config: Option<PathBuf>,

    #[arg(short, long)]
    pub verbose: bool,

    #[arg(long)]
    pub log_file: Option<PathBuf>,

    #[arg(long)]
    pub threads: Option<usize>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Analyze genomic sequences
    Analyze(AnalyzeArgs),
    
    /// Train AI models
    Train(TrainArgs),
    
    /// Run model inference
    Infer(InferArgs),
    
    /// Manage models and data
    Manage(ManageArgs),
    
    /// Start API server
    Serve(ServeArgs),
}

#[derive(Args)]
pub struct AnalyzeArgs {
    /// Input sequence file(s)
    #[arg(required = true)]
    pub input: Vec<PathBuf>,

    /// Output directory
    #[arg(short, long)]
    pub output: PathBuf,

    /// Sequence format
    #[arg(short, long, default_value = "auto")]
    pub format: SequenceFormat,

    /// Analysis configuration file
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Minimum sequence quality
    #[arg(long)]
    pub min_quality: Option<f64>,

    /// Reference genome
    #[arg(long)]
    pub reference: Option<PathBuf>,

    /// Number of threads for processing
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// Compression level (0-9)
    #[arg(short, long)]
    pub compression: Option<u32>,

    /// Write intermediate files
    #[arg(long)]
    pub keep_intermediate: bool,
}

#[derive(Args)]
pub struct TrainArgs {
    /// Training data directory
    #[arg(short, long)]
    pub data: PathBuf,

    /// Model configuration file
    #[arg(short, long)]
    pub model_config: PathBuf,

    /// Training configuration file
    #[arg(short, long)]
    pub train_config: Option<PathBuf>,

    /// Output directory for checkpoints
    #[arg(short, long)]
    pub output: PathBuf,

    /// Resume from checkpoint
    #[arg(long)]
    pub resume: bool,

    /// Enable mixed precision training
    #[arg(long)]
    pub mixed_precision: bool,

    /// Number of GPUs to use
    #[arg(long)]
    pub num_gpus: Option<usize>,

    /// Distributed training
    #[arg(long)]
    pub distributed: bool,

    /// Training epochs
    #[arg(short, long)]
    pub epochs: Option<usize>,
}

#[derive(Args)]
pub struct InferArgs {
    /// Model ID or path
    #[arg(short, long)]
    pub model: String,

    /// Input data file(s)
    #[arg(required = true)]
    pub input: Vec<PathBuf>,

    /// Output directory
    #[arg(short, long)]
    pub output: PathBuf,

    /// Batch size
    #[arg(short, long)]
    pub batch_size: Option<usize>,

    /// Use GPU acceleration
    #[arg(long)]
    pub gpu: bool,

    /// Quantization (none, int8, fp16)
    #[arg(long)]
    pub quantize: Option<String>,

    /// Maximum concurrent requests
    #[arg(long)]
    pub max_concurrent: Option<usize>,
}

#[derive(Args)]
pub struct ManageArgs {
    #[command(subcommand)]
    pub command: ManageCommands,
}

#[derive(Args)]
pub struct ServeArgs {
    /// Host address
    #[arg(short, long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port number
    #[arg(short, long, default_value_t = 8080)]
    pub port: u16,

    /// TLS certificate file
    #[arg(long)]
    pub cert: Option<PathBuf>,

    /// TLS key file
    #[arg(long)]
    pub key: Option<PathBuf>,

    /// Number of worker threads
    #[arg(long)]
    pub workers: Option<usize>,

    /// Enable development mode
    #[arg(long)]
    pub dev: bool,
}

#[derive(Subcommand)]
pub enum ManageCommands {
    /// List available models
    List {
        #[arg(short, long)]
        format: Option<String>,
    },
    
    /// Download a model
    Download {
        model_id: String,
        #[arg(short, long)]
        version: Option<String>,
    },
    
    /// Upload a model
    Upload {
        path: PathBuf,
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        description: Option<String>,
    },
    
    /// Delete a model
    Delete {
        model_id: String,
        #[arg(long)]
        force: bool,
    },
}

impl Commands {
    pub async fn execute(self, global_config: GlobalConfig) -> Result<()> {
        match self {
            Self::Analyze(args) => execute_analyze(args, global_config).await,
            Self::Train(args) => execute_train(args, global_config).await,
            Self::Infer(args) => execute_infer(args, global_config).await,
            Self::Manage(args) => execute_manage(args, global_config).await,
            Self::Serve(args) => execute_serve(args, global_config).await,
        }
    }
}

async fn execute_analyze(args: AnalyzeArgs, config: GlobalConfig) -> Result<()> {
    info!("Starting sequence analysis");

    let analysis_config = if let Some(config_path) = args.config {
        AnalysisConfig::from_file(&config_path).await?
    } else {
        AnalysisConfig::default()
    };

    // Create output directory
    fs::create_dir_all(&args.output).await?;

    // Process each input file
    for input in args.input {
        info!("Processing file: {}", input.display());
        
        let result = analyze_sequence(
            &input,
            &args.output,
            &analysis_config,
            args.format,
            args.min_quality,
        ).await;

        match result {
            Ok(_) => info!("Successfully processed {}", input.display()),
            Err(e) => {
                error!("Failed to process {}: {}", input.display(), e);
                if !config.continue_on_error {
                    return Err(e);
                }
            }
        }
    }

    Ok(())
}

async fn execute_train(args: TrainArgs, config: GlobalConfig) -> Result<()> {
    info!("Starting model training");

    let model_config = ModelConfig::from_file(&args.model_config).await?;
    let train_config = if let Some(config_path) = args.train_config {
        TrainingConfig::from_file(&config_path).await?
    } else {
        TrainingConfig::default()
    };

    // Set up distributed training if enabled
    if args.distributed {
        setup_distributed_training(&args, &config).await?;
    }

    // Initialize training
    let trainer = initialize_trainer(
        model_config,
        train_config,
        args.mixed_precision,
        args.num_gpus,
    ).await?;

    // Start training
    trainer.train(
        &args.data,
        &args.output,
        args.epochs,
        args.resume,
    ).await?;

    info!("Training completed successfully");
    Ok(())
}

async fn execute_infer(args: InferArgs, config: GlobalConfig) -> Result<()> {
    info!("Starting model inference");

    let inference_config = InferenceConfig {
        batch_size: args.batch_size,
        use_gpu: args.gpu,
        quantization: args.quantize.map(|q| q.parse().unwrap()),
        max_concurrent: args.max_concurrent,
    };

    // Load model
    let model = load_model(&args.model, &inference_config).await?;

    // Process inputs
    let mut results = Vec::new();
    for input in args.input {
        let result = model.infer(&input).await?;
        results.push(result);
    }

    // Save results
    save_inference_results(&args.output, &results).await?;

    info!("Inference completed successfully");
    Ok(())
}

async fn execute_manage(args: ManageArgs, config: GlobalConfig) -> Result<()> {
    match args.command {
        ManageCommands::List { format } => {
            list_models(format.as_deref()).await?;
        },
        ManageCommands::Download { model_id, version } => {
            download_model(&model_id, version.as_deref()).await?;
        },
        ManageCommands::Upload { path, name, description } => {
            upload_model(&path, &name, description.as_deref()).await?;
        },
        ManageCommands::Delete { model_id, force } => {
            delete_model(&model_id, force).await?;
        },
    }
    Ok(())
}

async fn execute_serve(args: ServeArgs, config: GlobalConfig) -> Result<()> {
    info!("Starting API server on {}:{}", args.host, args.port);

    let server_config = ServerConfig {
        host: args.host,
        port: args.port,
        tls_config: if let (Some(cert), Some(key)) = (args.cert, args.key) {
            Some(TlsConfig { cert, key })
        } else {
            None
        },
        workers: args.workers,
        development_mode: args.dev,
    };

    run_server(server_config).await?;

    Ok(())
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

        let output = temp.child("output");
        
        let args = AnalyzeArgs {
            input: vec![input.path().to_owned()],
            output: output.path().to_owned(),
            format: SequenceFormat::Fasta,
            config: None,
            min_quality: None,
            reference: None,
            threads: None,
            compression: None,
            keep_intermediate: false,
        };

        let config = GlobalConfig::default();
        execute_analyze(args, config).await.unwrap();

        output.child("results.json").assert(predicate::path::exists());
    }

    #[tokio::test]
    async fn test_train_command() {
        let temp = assert_fs::TempDir::new().unwrap();
        let config_file = temp.child("model_config.json");
        config_file.write_str(r#"{"type": "transformer"}"#).unwrap();

        let args = TrainArgs {
            data: temp.path().to_owned(),
            model_config: config_file.path().to_owned(),
            train_config: None,
            output: temp.path().to_owned(),
            resume: false,
            mixed_precision: false,
            num_gpus: None,
            distributed: false,
            epochs: Some(1),
        };

        let config = GlobalConfig::default();
        execute_train(args, config).await.unwrap();
    }
}