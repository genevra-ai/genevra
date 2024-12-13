use std::{
    fmt::{self, Display},
    error::Error as StdError,
    io,
    sync::Arc,
    path::{Path, PathBuf},
    collections::HashMap,
    num::ParseIntError,
    time::{SystemTime, Duration},
};

use backtrace::Backtrace;
use thiserror::Error;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {source}")]
    Io {
        #[from]
        source: io::Error,
        backtrace: Backtrace,
    },

    #[error("Genomics error: {kind}")]
    Genomics {
        kind: GenomicsErrorKind,
        backtrace: Backtrace,
    },

    #[error("AI error: {kind}")]
    AI {
        kind: AIErrorKind,
        backtrace: Backtrace,
    },

    #[error("Configuration error: {kind}")]
    Config {
        kind: ConfigErrorKind,
        backtrace: Backtrace,
    },

    #[error("Resource error: {kind}")]
    Resource {
        kind: ResourceErrorKind,
        context: ErrorContext,
        backtrace: Backtrace,
    },

    #[error("Processing error: {kind}")]
    Processing {
        kind: ProcessingErrorKind,
        context: ErrorContext,
        backtrace: Backtrace,
    },

    #[error("Network error: {kind}")]
    Network {
        kind: NetworkErrorKind,
        context: ErrorContext,
        backtrace: Backtrace,
    },

    #[error("Security error: {kind}")]
    Security {
        kind: SecurityErrorKind,
        context: ErrorContext,
        backtrace: Backtrace,
    },

    #[error("Validation error: {details}")]
    Validation {
        details: String,
        violations: Vec<ValidationViolation>,
        backtrace: Backtrace,
    },

    #[error("External service error: {service} - {details}")]
    ExternalService {
        service: String,
        details: String,
        retry_after: Option<Duration>,
        backtrace: Backtrace,
    },

    #[error("Rate limit exceeded: {details}")]
    RateLimit {
        details: String,
        limit: u32,
        reset_after: Duration,
        backtrace: Backtrace,
    },

    #[error("Internal error: {0}")]
    Internal(String, Backtrace),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum GenomicsErrorKind {
    #[error("Invalid sequence format: {0}")]
    InvalidSequenceFormat(String),
    #[error("Sequence alignment failed: {0}")]
    AlignmentFailed(String),
    #[error("Invalid quality scores: {0}")]
    InvalidQualityScores(String),
    #[error("Reference genome error: {0}")]
    ReferenceGenomeError(String),
    #[error("Variant calling failed: {0}")]
    VariantCallingFailed(String),
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum AIErrorKind {
    #[error("Model initialization failed: {0}")]
    ModelInitFailed(String),
    #[error("Invalid model architecture: {0}")]
    InvalidArchitecture(String),
    #[error("Training error: {0}")]
    TrainingError(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("GPU error: {0}")]
    GPUError(String),
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum ConfigErrorKind {
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Invalid value for {field}: {details}")]
    InvalidValue { field: String, details: String },
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum ResourceErrorKind {
    #[error("Resource not found: {resource_type}/{resource_id}")]
    NotFound {
        resource_type: String,
        resource_id: String,
    },
    #[error("Resource locked: {resource_type}/{resource_id}")]
    Locked {
        resource_type: String,
        resource_id: String,
        locked_by: String,
        lock_expiry: DateTime<Utc>,
    },
    #[error("Resource quota exceeded: {resource_type}")]
    QuotaExceeded {
        resource_type: String,
        current: u64,
        maximum: u64,
    },
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum ProcessingErrorKind {
    #[error("Pipeline error: {stage} - {details}")]
    PipelineError { stage: String, details: String },
    #[error("Data validation failed: {0}")]
    ValidationFailed(String),
    #[error("Processing timeout after {duration:?}")]
    Timeout { duration: Duration },
    #[error("Task queue full")]
    QueueFull,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum NetworkErrorKind {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    #[error("TLS error: {0}")]
    TlsError(String),
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum SecurityErrorKind {
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    #[error("Authorization failed: {0}")]
    AuthorizationFailed(String),
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Token expired")]
    TokenExpired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    timestamp: DateTime<Utc>,
    request_id: Option<String>,
    user_id: Option<String>,
    operation: Option<String>,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    field: String,
    code: String,
    message: String,
    path: Option<String>,
    value: Option<serde_json::Value>,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::Network { .. } |
            Error::ExternalService { .. } |
            Error::RateLimit { .. } |
            Error::Resource { 
                kind: ResourceErrorKind::Locked { .. }, 
                ..
            }
        )
    }

    pub fn status_code(&self) -> u16 {
        match self {
            Error::Validation { .. } => 400,
            Error::Security { kind: SecurityErrorKind::AuthenticationFailed(_), .. } => 401,
            Error::Security { kind: SecurityErrorKind::AuthorizationFailed(_), .. } => 403,
            Error::Resource { kind: ResourceErrorKind::NotFound { .. }, .. } => 404,
            Error::Resource { kind: ResourceErrorKind::Locked { .. }, .. } => 423,
            Error::RateLimit { .. } => 429,
            _ => 500,
        }
    }

    pub fn error_code(&self) -> String {
        match self {
            Error::Genomics { kind, .. } => format!("GENOMICS_{:?}", kind),
            Error::AI { kind, .. } => format!("AI_{:?}", kind),
            Error::Config { kind, .. } => format!("CONFIG_{:?}", kind),
            Error::Resource { kind, .. } => format!("RESOURCE_{:?}", kind),
            Error::Processing { kind, .. } => format!("PROCESSING_{:?}", kind),
            Error::Network { kind, .. } => format!("NETWORK_{:?}", kind),
            Error::Security { kind, .. } => format!("SECURITY_{:?}", kind),
            Error::Validation { .. } => "VALIDATION_ERROR".to_string(),
            Error::ExternalService { .. } => "EXTERNAL_SERVICE_ERROR".to_string(),
            Error::RateLimit { .. } => "RATE_LIMIT_ERROR".to_string(),
            Error::Internal(..) => "INTERNAL_ERROR".to_string(),
            Error::Io { .. } => "IO_ERROR".to_string(),
            Error::Other(_) => "UNKNOWN_ERROR".to_string(),
        }
    }
}

pub struct ErrorTracker {
    errors: Arc<RwLock<Vec<TrackedError>>>,
}

#[derive(Debug, Clone, Serialize)]
struct TrackedError {
    error_code: String,
    message: String,
    context: ErrorContext,
    timestamp: DateTime<Utc>,
    count: u64,
}

impl ErrorTracker {
    pub fn new() -> Self {
        Self {
            errors: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn track(&self, error: &Error) {
        let tracked = TrackedError {
            error_code: error.error_code(),
            message: error.to_string(),
            context: self.extract_context(error),
            timestamp: Utc::now(),
            count: 1,
        };

        let mut errors = self.errors.write().await;
        if let Some(existing) = errors.iter_mut()
            .find(|e| e.error_code == tracked.error_code) {
            existing.count += 1;
        } else {
            errors.push(tracked);
        }
    }

    fn extract_context(&self, error: &Error) -> ErrorContext {
        match error {
            Error::Resource { context, .. } |
            Error::Processing { context, .. } |
            Error::Network { context, .. } |
            Error::Security { context, .. } => context.clone(),
            _ => ErrorContext {
                timestamp: Utc::now(),
                request_id: None,
                user_id: None,
                operation: None,
                metadata: HashMap::new(),
            },
        }
    }
}

impl Default for ErrorTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_error_is_retryable() {
        let network_error = Error::Network {
            kind: NetworkErrorKind::ConnectionFailed("connection refused".to_string()),
            context: ErrorContext {
                timestamp: Utc::now(),
                request_id: None,
                user_id: None,
                operation: None,
                metadata: HashMap::new(),
            },
            backtrace: Backtrace::capture(),
        };
        assert!(network_error.is_retryable());

        let validation_error = Error::Validation {
            details: "invalid input".to_string(),
            violations: vec![],
            backtrace: Backtrace::capture(),
        };
        assert!(!validation_error.is_retryable());
    }

    #[tokio::test]
    async fn test_error_tracking() {
        let tracker = ErrorTracker::new();
        let error = Error::Processing {
            kind: ProcessingErrorKind::Timeout {
                duration: Duration::from_secs(30),
            },
            context: ErrorContext {
                timestamp: Utc::now(),
                request_id: Some("req-123".to_string()),
                user_id: None,
                operation: Some("process_data".to_string()),
                metadata: HashMap::new(),
            },
            backtrace: Backtrace::capture(),
        };

        tracker.track(&error).await;
        tracker.track(&error).await;

        let errors = tracker.errors.read().await;
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].count, 2);
    }
}