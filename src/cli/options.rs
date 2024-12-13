use std::{
    path::{Path, PathBuf},
    str::FromStr,
    time::Duration,
    num::NonZeroUsize,
};

use anyhow::{Result, Context, bail};
use serde::{Serialize, Deserialize};
use clap::Args;

#[derive(Debug, Clone, Args)]
pub struct CommonOptions {
    /// Path to configuration file
    #[arg(long, env = "GENOME_AI_CONFIG")]
    pub config: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long, env = "GENOME_AI_VERBOSE")]
    pub verbose: bool,

    /// Number of threads to use
    #[arg(long, env = "GENOME_AI_THREADS")]
    pub threads: Option<NonZeroUsize>,

    /// Output format (json, text, table)
    #[arg(long, value_parser = parse_output_format)]
    pub output_format: Option<OutputFormat>,

    /// Operation timeout in seconds
    #[arg(long)]
    pub timeout: Option<u64>,

    /// Write logs to file
    #[arg(long)]
    pub log_file: Option<PathBuf>,

    /// Continue on errors
    #[arg(long)]
    pub continue_on_error: bool,
}

#[derive(Debug, Clone, Args)]
pub struct AnalysisOptions {
    /// Minimum sequence quality score
    #[arg(long)]
    pub min_quality: Option<f64>,

    /// Coverage threshold
    #[arg(long)]
    pub min_coverage: Option<u32>,

    /// Maximum error rate
    #[arg(long)]
    pub max_error_rate: Option<f64>,

    /// Reference genome path
    #[arg(long)]
    pub reference: Option<PathBuf>,

    /// Sequence format (auto, fasta, fastq, bam)
    #[arg(long, value_parser = parse_sequence_format)]
    pub format: Option<SequenceFormat>,

    /// Enable variant calling
    #[arg(long)]
    pub call_variants: bool,

    /// Compression level (0-9)
    #[arg(long)]
    pub compression: Option<u32>,

    /// Keep intermediate files
    #[arg(long)]
    pub keep_intermediate: bool,
}

#[derive(Debug, Clone, Args)]
pub struct ModelOptions {
    /// Model architecture (transformer, cnn, hybrid)
    #[arg(long)]
    pub architecture: Option<ModelArchitecture>,

    /// Model checkpoint path
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,

    /// Batch size for processing
    #[arg(long)]
    pub batch_size: Option<NonZeroUsize>,

    /// Enable mixed precision
    #[arg(long)]
    pub mixed_precision: bool,

    /// Use GPU acceleration
    #[arg(long)]
    pub gpu: bool,

    /// Model quantization (none, int8, fp16)
    #[arg(long)]
    pub quantization: Option<QuantizationType>,
}

#[derive(Debug, Clone, Args)]
pub struct ServerOptions {
    /// Server host address
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Server port
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Enable TLS/HTTPS
    #[arg(long)]
    pub tls: bool,

    /// TLS certificate path
    #[arg(long)]
    pub cert: Option<PathBuf>,

    /// TLS key path
    #[arg(long)]
    pub key: Option<PathBuf>,

    /// Number of worker threads
    #[arg(long)]
    pub workers: Option<NonZeroUsize>,

    /// Enable development mode
    #[arg(long)]
    pub dev: bool,

    /// Rate limit (requests per second)
    #[arg(long)]
    pub rate_limit: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Json,
    Text,
    Table,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SequenceFormat {
    Auto,
    Fasta,
    Fastq,
    Bam,
    Sam,
    Vcf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    Transformer,
    CNN,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationType {
    None,
    Int8,
    Fp16,
    Dynamic,
}

impl CommonOptions {
    pub fn validate(&self) -> Result<()> {
        if let Some(ref config) = self.config {
            if !config.exists() {
                bail!("Config file does not exist: {}", config.display());
            }
        }

        if let Some(ref log_file) = self.log_file {
            if let Some(parent) = log_file.parent() {
                if !parent.exists() {
                    bail!("Log file directory does not exist: {}", parent.display());
                }
            }
        }

        Ok(())
    }

    pub fn get_timeout(&self) -> Duration {
        Duration::from_secs(self.timeout.unwrap_or(3600))
    }
}

impl AnalysisOptions {
    pub fn validate(&self) -> Result<()> {
        if let Some(min_quality) = self.min_quality {
            if !(0.0..=100.0).contains(&min_quality) {
                bail!("Min quality must be between 0 and 100");
            }
        }

        if let Some(max_error_rate) = self.max_error_rate {
            if !(0.0..=1.0).contains(&max_error_rate) {
                bail!("Error rate must be between 0 and 1");
            }
        }

        if let Some(ref reference) = self.reference {
            if !reference.exists() {
                bail!("Reference genome does not exist: {}", reference.display());
            }
        }

        if let Some(compression) = self.compression {
            if compression > 9 {
                bail!("Compression level must be between 0 and 9");
            }
        }

        Ok(())
    }

    pub fn detect_format(&self, path: &Path) -> Result<SequenceFormat> {
        if let Some(format) = self.format {
            return Ok(format);
        }

        match path.extension().and_then(|ext| ext.to_str()) {
            Some("fa") | Some("fasta") => Ok(SequenceFormat::Fasta),
            Some("fq") | Some("fastq") => Ok(SequenceFormat::Fastq),
            Some("bam") => Ok(SequenceFormat::Bam),
            Some("sam") => Ok(SequenceFormat::Sam),
            Some("vcf") => Ok(SequenceFormat::Vcf),
            _ => Ok(SequenceFormat::Auto),
        }
    }
}

impl ModelOptions {
    pub fn validate(&self) -> Result<()> {
        if let Some(ref checkpoint) = self.checkpoint {
            if !checkpoint.exists() {
                bail!("Checkpoint file does not exist: {}", checkpoint.display());
            }
        }

        if self.gpu && !cuda_available() {
            bail!("GPU acceleration requested but CUDA is not available");
        }

        Ok(())
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size.map_or(32, |bs| bs.get())
    }
}

impl ServerOptions {
    pub fn validate(&self) -> Result<()> {
        if self.tls {
            let cert = self.cert.as_ref().context("TLS enabled but no certificate provided")?;
            let key = self.key.as_ref().context("TLS enabled but no key provided")?;

            if !cert.exists() {
                bail!("TLS certificate does not exist: {}", cert.display());
            }
            if !key.exists() {
                bail!("TLS key does not exist: {}", key.display());
            }
        }

        if let Some(rate_limit) = self.rate_limit {
            if rate_limit == 0 {
                bail!("Rate limit must be greater than 0");
            }
        }

        Ok(())
    }

    pub fn get_workers(&self) -> usize {
        self.workers.map_or_else(num_cpus::get, |w| w.get())
    }
}

fn parse_output_format(s: &str) -> Result<OutputFormat> {
    match s.to_lowercase().as_str() {
        "json" => Ok(OutputFormat::Json),
        "text" => Ok(OutputFormat::Text),
        "table" => Ok(OutputFormat::Table),
        _ => bail!("Invalid output format: {}", s),
    }
}

fn parse_sequence_format(s: &str) -> Result<SequenceFormat> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(SequenceFormat::Auto),
        "fasta" => Ok(SequenceFormat::Fasta),
        "fastq" => Ok(SequenceFormat::Fastq),
        "bam" => Ok(SequenceFormat::Bam),
        "sam" => Ok(SequenceFormat::Sam),
        "vcf" => Ok(SequenceFormat::Vcf),
        _ => bail!("Invalid sequence format: {}", s),
    }
}

fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        tch::Cuda::is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_common_options_validation() {
        let temp = tempdir().unwrap();
        let config = temp.path().join("config.json");
        std::fs::write(&config, "{}").unwrap();

        let options = CommonOptions {
            config: Some(config),
            verbose: false,
            threads: None,
            output_format: None,
            timeout: Some(60),
            log_file: None,
            continue_on_error: false,
        };

        assert!(options.validate().is_ok());
    }

    #[test]
    fn test_analysis_options_validation() {
        let options = AnalysisOptions {
            min_quality: Some(50.0),
            min_coverage: Some(10),
            max_error_rate: Some(0.1),
            reference: None,
            format: Some(SequenceFormat::Fasta),
            call_variants: false,
            compression: Some(5),
            keep_intermediate: false,
        };

        assert!(options.validate().is_ok());

        let invalid_options = AnalysisOptions {
            min_quality: Some(150.0),
            ..options
        };

        assert!(invalid_options.validate().is_err());
    }

    #[test]
    fn test_model_options_validation() {
        let options = ModelOptions {
            architecture: Some(ModelArchitecture::Transformer),
            checkpoint: None,
            batch_size: NonZeroUsize::new(32),
            mixed_precision: false,
            gpu: false,
            quantization: Some(QuantizationType::None),
        };

        assert!(options.validate().is_ok());
    }

    #[test]
    fn test_server_options_validation() {
        let temp = tempdir().unwrap();
        let cert = temp.path().join("cert.pem");
        let key = temp.path().join("key.pem");
        std::fs::write(&cert, "").unwrap();
        std::fs::write(&key, "").unwrap();

        let options = ServerOptions {
            host: "127.0.0.1".to_string(),
            port: 8080,
            tls: true,
            cert: Some(cert),
            key: Some(key),
            workers: None,
            dev: false,
            rate_limit: Some(1000),
        };

        assert!(options.validate().is_ok());
    }

    #[test]
    fn test_format_parsing() {
        assert_eq!(parse_output_format("json").unwrap(), OutputFormat::Json);
        assert_eq!(parse_output_format("TEXT").unwrap(), OutputFormat::Text);
        assert!(parse_output_format("invalid").is_err());

        assert_eq!(parse_sequence_format("fasta").unwrap(), SequenceFormat::Fasta);
        assert_eq!(parse_sequence_format("BAM").unwrap(), SequenceFormat::Bam);
        assert!(parse_sequence_format("invalid").is_err());
    }

    #[test]
    fn test_format_detection() {
        let options = AnalysisOptions {
            format: None,
            ..Default::default()
        };

        assert_eq!(
            options.detect_format(Path::new("test.fa")).unwrap(),
            SequenceFormat::Fasta
        );
        assert_eq!(
            options.detect_format(Path::new("test.bam")).unwrap(),
            SequenceFormat::Bam
        );
    }
}