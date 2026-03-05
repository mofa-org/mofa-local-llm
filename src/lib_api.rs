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
        
        config.save()
            .map_err(|e| format!("Failed to save config: {}", e))?;

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
        // Check if already loaded (quick read check first)
        {
            let engines = self.engines.read()
                .map_err(|_| "Lock poisoned: another thread panicked while holding the lock".to_string())?;
            if engines.contains_key(model_id) {
                return Ok(());
            }
        }

        // Get model entry info (outside the write lock to minimize lock time)
        let model_entry = {
            let config = self.config.read()
                .map_err(|_| "Lock poisoned: another thread panicked while holding the lock".to_string())?;
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

        // Double-check and insert with write lock to prevent race condition
        let mut engines = self.engines.write()
            .map_err(|_| "Lock poisoned: another thread panicked while holding the lock".to_string())?;
        // Check again under write lock to prevent duplicate loads
        if engines.contains_key(model_id) {
            return Ok(());
        }

        // Load the model (expensive operation, but now protected by write lock)
        let engine = LlmEngine::load(model_path, backend, model_id)
            .map_err(|e| format!("Failed to load model '{}': {}", model_id, e))?;

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
        let engines = self.engines.read()
            .map_err(|_| "Lock poisoned: another thread panicked while holding the lock".to_string())?;
        let engine_arc = engines
            .get(model_id)
            .ok_or_else(|| format!("Model '{}' is not loaded. Call load_model() first.", model_id))?
            .clone();
        drop(engines);

        let mut engine = engine_arc.write()
            .map_err(|_| "Lock poisoned: another thread panicked while holding the lock".to_string())?;
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

    pub fn scan_models_dir(&self) -> Result<(), String> {
        let mut config = self.config.write()
            .map_err(|_| "Lock poisoned: another thread panicked while holding the lock".to_string())?;
        config.scan_models_dir();
        config.save()
            .map_err(|e| format!("Failed to save config: {}", e))?;
        Ok(())
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
            let engines = self.engines.read()
                .map_err(|_| "Lock poisoned: another thread panicked while holding the lock".to_string())?;
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

        let mut engines = self.engines.write()
            .map_err(|_| "Lock poisoned: another thread panicked while holding the lock".to_string())?;
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
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_new_with_default_dir() {
        let client = LocalLLMClient::new(None);
        assert!(client.is_ok());
    }

    #[test]
    fn test_new_with_custom_dir() {
        let temp_dir = std::env::temp_dir().join("test_models_mofa");
        let client = LocalLLMClient::new(Some(temp_dir));
        assert!(client.is_ok());
    }

    #[test]
    fn test_list_models() {
        let client = LocalLLMClient::new(None).unwrap();
        let models = client.list_models();
        // Verify all model IDs are non-empty strings
        assert!(models.iter().all(|id| !id.is_empty()));
    }

    #[test]
    fn test_is_model_loaded_false() {
        let client = LocalLLMClient::new(None).unwrap();
        assert!(!client.is_model_loaded("nonexistent-model"));
    }

    #[test]
    fn test_get_model_info_nonexistent() {
        let client = LocalLLMClient::new(None).unwrap();
        let info = client.get_model_info("nonexistent-model");
        assert!(info.is_none());
    }

    #[test]
    fn test_models_dir() {
        let client = LocalLLMClient::new(None).unwrap();
        let dir = client.models_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_loaded_models_empty() {
        let client = LocalLLMClient::new(None).unwrap();
        let loaded = client.loaded_models();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_unload_model_not_loaded() {
        let client = LocalLLMClient::new(None).unwrap();
        // Should not panic
        client.unload_model("nonexistent-model");
    }

    #[test]
    fn test_load_model_not_found() {
        let client = LocalLLMClient::new(None).unwrap();
        let result = client.load_model("nonexistent-model");
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("not found") || error_msg.contains("Model"));
    }

    #[test]
    fn test_chat_model_not_loaded() {
        let client = LocalLLMClient::new(None).unwrap();
        let messages = vec![("user".to_string(), "test".to_string())];
        let result = client.chat("nonexistent-model", &messages, 10, 0.7);
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("not loaded") || error_msg.contains("Model"));
    }

    #[test]
    fn test_clone() {
        let client = LocalLLMClient::new(None).unwrap();
        let cloned = client.clone();
        assert_eq!(client.list_models(), cloned.list_models());
    }

    #[test]
    fn test_concurrent_list_models() {
        let client = Arc::new(LocalLLMClient::new(None).unwrap());
        let mut handles = vec![];

        for _ in 0..10 {
            let client_clone = Arc::clone(&client);
            handles.push(thread::spawn(move || {
                client_clone.list_models()
            }));
        }

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // All threads should get the same result
        assert!(results.iter().all(|r| r == &results[0]));
    }

    #[test]
    fn test_concurrent_is_model_loaded() {
        let client = Arc::new(LocalLLMClient::new(None).unwrap());
        let mut handles = vec![];

        for _ in 0..10 {
            let client_clone = Arc::clone(&client);
            handles.push(thread::spawn(move || {
                client_clone.is_model_loaded("test-model")
            }));
        }

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // All threads should get the same result
        assert!(results.iter().all(|r| r == &results[0]));
    }

    #[test]
    fn test_concurrent_get_model_info() {
        let client = Arc::new(LocalLLMClient::new(None).unwrap());
        let mut handles = vec![];

        for _ in 0..10 {
            let client_clone = Arc::clone(&client);
            handles.push(thread::spawn(move || {
                client_clone.get_model_info("test-model")
            }));
        }

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // All threads should get the same result
        assert!(results.iter().all(|r| r == &results[0]));
    }

    #[test]
    fn test_scan_models_dir() {
        let client = LocalLLMClient::new(None).unwrap();
        // Should not panic
        let result = client.scan_models_dir();
        assert!(result.is_ok());
    }

    #[test]
    fn test_config() {
        let client = LocalLLMClient::new(None).unwrap();
        let config = client.config();
        assert!(!config.models_dir.is_empty());
    }

    #[test]
    fn test_default_model() {
        let client = LocalLLMClient::new(None).unwrap();
        let default = client.default_model();
        // May be None if no models available
        if let Some(model_id) = default {
            assert!(!model_id.is_empty());
        }
    }

    #[test]
    fn test_is_model_available() {
        let client = LocalLLMClient::new(None).unwrap();
        let available = client.is_model_available("nonexistent-model");
        assert!(!available);
    }

    #[test]
    fn test_get_model_backend_nonexistent() {
        let client = LocalLLMClient::new(None).unwrap();
        let backend = client.get_model_backend("nonexistent-model");
        assert!(backend.is_none());
    }

    #[test]
    fn test_load_model_from_path_invalid() {
        let client = LocalLLMClient::new(None).unwrap();
        let invalid_path = std::path::Path::new("/nonexistent/path/to/model");
        let result = client.load_model_from_path(invalid_path, None);
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("does not exist"));
    }
}
