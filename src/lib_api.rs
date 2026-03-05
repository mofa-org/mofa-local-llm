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

    pub fn get_model_info(&self, model_id: &str) -> Option<ModelEntry> {
        let config = self.config.read().unwrap();
        config
            .models
            .iter()
            .find(|m| m.id == model_id)
            .cloned()
    }

    pub fn load_model(&self, model_id: &str) -> Result<(), String> {
        {
            let engines = self.engines.read().unwrap();
            if engines.contains_key(model_id) {
                return Ok(());
            }
        }

        let model_entry = {
            let config = self.config.read().unwrap();
            config
                .models
                .iter()
                .find(|m| m.id == model_id)
                .cloned()
                .ok_or_else(|| format!("Model '{}' not found. Use list_models() to see available models.", model_id))?
        };

        let backend = detect_backend(&model_entry.model_type)
            .ok_or_else(|| format!("Unknown model type: {}", model_entry.model_type))?;

        let model_path = Path::new(&model_entry.path);
        if !model_path.exists() {
            return Err(format!("Model path does not exist: {}", model_entry.path));
        }

        let engine = LlmEngine::load(model_path, backend, model_id)
            .map_err(|e| format!("Failed to load model '{}': {}", model_id, e))?;

        let mut engines = self.engines.write().unwrap();
        engines.insert(model_id.to_string(), Arc::new(RwLock::new(engine)));

        Ok(())
    }

    pub fn chat(
        &self,
        model_id: &str,
        messages: &[(String, String)],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<ChatResult, String> {
        let engines = self.engines.read().unwrap();
        let engine_arc = engines
            .get(model_id)
            .ok_or_else(|| format!("Model '{}' is not loaded. Call load_model() first.", model_id))?
            .clone();
        drop(engines);

        let mut engine = engine_arc.write().unwrap();
        engine.chat(messages, max_tokens, temperature)
    }

    pub fn chat_with_auto_load(
        &self,
        model_id: &str,
        messages: &[(String, String)],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<ChatResult, String> {
        if !self.is_model_loaded(model_id) {
            self.load_model(model_id)?;
        }
        self.chat(model_id, messages, max_tokens, temperature)
    }

    pub fn unload_model(&self, model_id: &str) {
        let mut engines = self.engines.write().unwrap();
        engines.remove(model_id);
    }

    pub fn scan_models_dir(&self) {
        let mut config = self.config.write().unwrap();
        config.scan_models_dir();
        let _ = config.save();
    }

    pub fn config(&self) -> AppConfig {
        self.config.read().unwrap().clone()
    }

    pub fn loaded_models(&self) -> Vec<String> {
        let engines = self.engines.read().unwrap();
        engines.keys().cloned().collect()
    }

    pub fn default_model(&self) -> Option<String> {
        let models = self.list_models();
        models.first().cloned()
    }

    pub fn is_model_available(&self, model_id: &str) -> bool {
        let config = self.config.read().unwrap();
        config
            .models
            .iter()
            .any(|m| m.id == model_id && Path::new(&m.path).exists())
    }

    pub fn get_model_backend(&self, model_id: &str) -> Option<LlmBackend> {
        let model_entry = self.get_model_info(model_id)?;
        detect_backend(&model_entry.model_type)
    }

    pub fn load_model_from_path(
        &self,
        model_path: impl AsRef<Path>,
        model_id: Option<&str>,
    ) -> Result<String, String> {
        let model_path = model_path.as_ref();
        if !model_path.exists() {
            return Err(format!("Model path does not exist: {}", model_path.display()));
        }

        let model_id = model_id
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                model_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            });

        {
            let engines = self.engines.read().unwrap();
            if engines.contains_key(&model_id) {
                return Ok(model_id);
            }
        }

        let config_json = model_path.join("config.json");
        let model_type = if config_json.exists() {
            std::fs::read_to_string(&config_json)
                .ok()
                .and_then(|s| {
                    serde_json::from_str::<serde_json::Value>(&s)
                        .ok()
                        .and_then(|v| v.get("model_type").and_then(|v| v.as_str()).map(|s| s.to_string()))
                })
                .unwrap_or_else(|| "qwen2".to_string())
        } else {
            "qwen2".to_string()
        };

        let backend = detect_backend(&model_type)
            .ok_or_else(|| format!("Unknown model type: {}", model_type))?;

        let engine = LlmEngine::load(model_path, backend, &model_id)
            .map_err(|e| format!("Failed to load model from path: {}", e))?;

        let mut engines = self.engines.write().unwrap();
        engines.insert(model_id.clone(), Arc::new(RwLock::new(engine)));

        Ok(model_id)
    }
}

impl Clone for LocalLLMClient {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            engines: Arc::clone(&self.engines),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_default_dir() {
        let client = LocalLLMClient::new(None);
        assert!(client.is_ok());
    }

    #[test]
    fn test_list_models() {
        let client = LocalLLMClient::new(None).unwrap();
        let models = client.list_models();
        assert!(models.len() >= 0);
    }

    #[test]
    fn test_is_model_loaded_false() {
        let client = LocalLLMClient::new(None).unwrap();
        assert!(!client.is_model_loaded("nonexistent-model"));
    }
}
