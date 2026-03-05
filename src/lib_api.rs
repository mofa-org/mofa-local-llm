//! Library API for mofa-local-llm
//!
//! This module provides a clean, thread-safe API for using mofa-local-llm
//! as a library, without requiring the HTTP server or GUI components.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::config::{AppConfig, ModelEntry};
use crate::inference::llm::{LlmBackend, LlmEngine};
use crate::inference::ChatResult;

/// Detects the LLM backend from a model type string.
fn detect_backend(model_type: &str) -> Option<LlmBackend> {
    match model_type {
        "qwen2" => Some(LlmBackend::Qwen2),
        "qwen3" | "qwen" => Some(LlmBackend::Qwen3),
        "mistral" => Some(LlmBackend::Mistral),
        "glm4" | "chatglm" => Some(LlmBackend::Glm4),
        "mixtral" => Some(LlmBackend::Mixtral),
        "minicpm" | "minicpm4" => Some(LlmBackend::MiniCpmSala),
        _ => None,
    }
}

/// Thread-safe client for managing and using local LLM models.
///
/// This client provides a library API for loading and using LLM models
/// without requiring the HTTP server. It manages model lifecycle and
/// provides thread-safe access to multiple models.
pub struct LocalLLMClient {
    config: Arc<RwLock<AppConfig>>,
    engines: Arc<RwLock<HashMap<String, Arc<RwLock<LlmEngine>>>>>,
}

impl LocalLLMClient {
    /// Create a new `LocalLLMClient` instance.
    pub fn new(models_dir: Option<PathBuf>) -> Result<Self, String> {
        let models_dir_str = models_dir
            .as_ref()
            .map(|p| p.to_string_lossy().to_string());

        let mut config = AppConfig::load(models_dir_str.as_deref());
        config.scan_models_dir();
        
        let _ = config.save();

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            engines: Arc::new(RwLock::new(HashMap::new())),
        })
    }
}
    pub fn models_dir(&self) -> PathBuf {
        let config = self.config.read().unwrap();
        PathBuf::from(&config.models_dir)
    }

    pub fn list_models(&self) -> Vec<String> {
        let config = self.config.read().unwrap();
        config.models.iter().map(|m| m.id.clone()).collect()
    }

    pub fn is_model_loaded(&self, model_id: &str) -> bool {
        let engines = self.engines.read().unwrap();
        engines.contains_key(model_id)
    }
}
