use crate::config::{calculate_model_size, AppConfig, ModelEntry, QuantInfo};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Download a model from HuggingFace to the local models directory.
pub fn download_model(
    repo_id: &str,
    models_dir: &Path,
    progress_cb: Option<&dyn Fn(&str)>,
) -> Result<ModelEntry, String> {
    let log = |msg: &str| {
        if let Some(cb) = &progress_cb {
            cb(msg);
        }
        eprintln!("[download] {}", msg);
    };

    // Resolve HF token
    let token = std::env::var("HF_TOKEN").ok().or_else(|| {
        let home = std::env::var("HOME").ok()?;
        let token_path = PathBuf::from(home).join(".cache/huggingface/token");
        std::fs::read_to_string(token_path)
            .ok()
            .map(|s| s.trim().to_string())
    });

    let api = if let Some(ref token) = token {
        hf_hub::api::sync::ApiBuilder::new()
            .with_token(Some(token.clone()))
            .build()
            .map_err(|e| format!("HF API error: {}", e))?
    } else {
        hf_hub::api::sync::ApiBuilder::new()
            .build()
            .map_err(|e| format!("HF API error: {}", e))?
    };

    let repo = api.model(repo_id.to_string());
    let model_id = repo_id.split('/').last().unwrap_or(repo_id);
    let dest_dir = models_dir.join(model_id);

    if dest_dir.exists() {
        // Check if it seems complete
        if dest_dir.join("config.json").exists() {
            return Err(format!("Model directory already exists: {}", dest_dir.display()));
        }
        // Incomplete download, remove and retry
        let _ = std::fs::remove_dir_all(&dest_dir);
    }

    std::fs::create_dir_all(&dest_dir)
        .map_err(|e| format!("Cannot create dir: {}", e))?;

    // Download essential files
    let essential_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ];
    for filename in &essential_files {
        log(&format!("Fetching {}...", filename));
        match repo.get(filename) {
            Ok(cached) => {
                std::fs::copy(&cached, dest_dir.join(filename))
                    .map_err(|e| format!("Failed to copy {}: {}", filename, e))?;
            }
            Err(e) => {
                // tokenizer_config.json is optional
                if *filename != "tokenizer_config.json" {
                    log(&format!("Warning: {} not found: {}", filename, e));
                }
            }
        }
    }

    // Download weights (sharded or single)
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        log("Found sharded weights, downloading...");
        std::fs::copy(&index_path, dest_dir.join("model.safetensors.index.json"))
            .map_err(|e| format!("Copy index error: {}", e))?;

        let index_content = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("Read index error: {}", e))?;
        let index: serde_json::Value =
            serde_json::from_str(&index_content).map_err(|e| format!("Parse index error: {}", e))?;

        if let Some(weight_map) = index["weight_map"].as_object() {
            let weight_files: HashSet<&str> =
                weight_map.values().filter_map(|v| v.as_str()).collect();
            let total = weight_files.len();
            for (i, weight_file) in weight_files.iter().enumerate() {
                log(&format!("Fetching weight {}/{}: {}...", i + 1, total, weight_file));
                let cached = repo
                    .get(weight_file)
                    .map_err(|e| format!("Download {} failed: {}", weight_file, e))?;
                std::fs::copy(&cached, dest_dir.join(weight_file))
                    .map_err(|e| format!("Copy {} failed: {}", weight_file, e))?;
            }
        }
    } else {
        log("Fetching model.safetensors...");
        let cached = repo
            .get("model.safetensors")
            .map_err(|e| format!("Download weights failed: {}", e))?;
        std::fs::copy(&cached, dest_dir.join("model.safetensors"))
            .map_err(|e| format!("Copy weights failed: {}", e))?;
    }

    // Try to download additional files that some models need
    for extra in &["am.mvn", "tokens.txt", "vocab.txt", "config.yaml"] {
        if let Ok(cached) = repo.get(extra) {
            let _ = std::fs::copy(&cached, dest_dir.join(extra));
        }
    }

    // Extract metadata
    let quant = detect_quantization(&dest_dir);
    let size_bytes = calculate_model_size(&dest_dir);
    let model_type = crate::config::detect_model_type(&dest_dir);

    let now = chrono::Utc::now().to_rfc3339();

    log(&format!("Download complete: {} ({} bytes)", model_id, size_bytes));

    Ok(ModelEntry {
        id: model_id.to_string(),
        repo_id: repo_id.to_string(),
        path: dest_dir.to_string_lossy().to_string(),
        model_type,
        quantization: quant,
        size_bytes: Some(size_bytes),
        downloaded_at: Some(now),
    })
}

fn detect_quantization(model_dir: &Path) -> Option<QuantInfo> {
    let config_path = model_dir.join("config.json");
    if let Ok(content) = std::fs::read_to_string(&config_path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(q) = v.get("quantization") {
                let bits = q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32;
                let group_size = q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
                return Some(QuantInfo { bits, group_size });
            }
        }
    }
    None
}



/// Register a downloaded model into the app config.
pub fn register_model(config: &mut AppConfig, entry: ModelEntry) {
    // Check if already registered
    if config.models.iter().any(|m| m.id == entry.id) {
        return;
    }
    config.models.push(entry);
    let _ = config.save();
}
