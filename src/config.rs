use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub models_dir: String,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default)]
    pub repo_id: String,
    pub path: String,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub quantization: Option<QuantInfo>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub downloaded_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantInfo {
    pub bits: i32,
    pub group_size: i32,
}

impl AppConfig {
    pub fn config_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".mofa")
    }

    pub fn config_path() -> PathBuf {
        Self::config_dir().join("ominix-config.json")
    }

    pub fn default_models_dir() -> String {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".mofa")
            .join("models")
            .to_string_lossy()
            .to_string()
    }

    pub fn load(models_dir_override: Option<&str>) -> Self {
        let path = Self::config_path();
        let mut config = if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_else(|| AppConfig {
                    models_dir: Self::default_models_dir(),
                    models: vec![],
                })
        } else {
            AppConfig {
                models_dir: Self::default_models_dir(),
                models: vec![],
            }
        };

        if let Some(dir) = models_dir_override {
            config.models_dir = dir.to_string();
        }

        config
    }

    pub fn save(&self) -> std::io::Result<()> {
        let dir = Self::config_dir();
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?;
        }
        let json = serde_json::to_string_pretty(self).unwrap();
        std::fs::write(Self::config_path(), json)
    }

    pub fn scan_models_dir(&mut self) {
        let models_dir = Path::new(&self.models_dir);
        if !models_dir.exists() {
            let _ = std::fs::create_dir_all(models_dir);
            return;
        }

        // Remove entries whose paths no longer exist
        self.models
            .retain(|entry| Path::new(&entry.path).exists());

        // Scan for new model subdirectories
        let known_paths: std::collections::HashSet<String> =
            self.models.iter().map(|m| m.path.clone()).collect();

        let entries = match std::fs::read_dir(models_dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let sub_path = entry.path();
            if !sub_path.is_dir() {
                continue;
            }

            let config_json = sub_path.join("config.json");
            if !config_json.exists() {
                continue;
            }

            let path_str = sub_path.to_string_lossy().to_string();
            if known_paths.contains(&path_str) {
                continue;
            }

            let size_bytes = calculate_model_size(&sub_path);
            let id = sub_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            // Try to detect model type from config.json
            let model_type = detect_model_type(&sub_path);

            self.models.push(ModelEntry {
                id,
                repo_id: String::new(),
                path: path_str,
                model_type,
                quantization: None,
                size_bytes: Some(size_bytes),
                downloaded_at: None,
            });
        }
    }
}

pub fn calculate_model_size(model_dir: &Path) -> u64 {
    std::fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "safetensors")
                .unwrap_or(false)
        })
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

pub fn detect_model_type(model_dir: &Path) -> String {
    let config_path = model_dir.join("config.json");
    if let Ok(content) = std::fs::read_to_string(&config_path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(mt) = v.get("model_type").and_then(|v| v.as_str()) {
                return mt.to_string();
            }
        }
    }
    "unknown".to_string()
}
