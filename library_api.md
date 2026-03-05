# Library API Documentation

This document describes the new library API added to mofa-local-llm for use as a dependency in other projects (e.g., MoFA).

## Overview

The library API provides a clean, thread-safe interface for loading and using LLM models without requiring the HTTP server or GUI components.

## Key Components

### `LocalLLMClient`

The main entry point for using mofa-local-llm as a library.

**Location**: `src/lib_api.rs`

**Exported from**: `mofa_local_llm::LocalLLMClient`

## Usage Example

```rust
use mofa_local_llm::LocalLLMClient;
use std::path::PathBuf;

// Create client with default model directory (~/.mofa/models/)
let client = LocalLLMClient::new(None)?;

// Or specify a custom models directory
let client = LocalLLMClient::new(Some(PathBuf::from("/path/to/models")))?;

// List available models
let models = client.list_models();
println!("Available models: {:?}", models);

// Load a model by ID
client.load_model("qwen2.5-0.5b-instruct")?;

// Or load from a direct path
let model_id = client.load_model_from_path("/path/to/model", None)?;

// Run inference
let result = client.chat(
    "qwen2.5-0.5b-instruct",
    &[
        ("system".to_string(), "You are a helpful assistant.".to_string()),
        ("user".to_string(), "Hello!".to_string()),
    ],
    128,  // max_tokens
    0.7,  // temperature
)?;

println!("Response: {}", result.text);
println!("Tokens: {}/{}", result.completion_tokens, result.prompt_tokens);
println!("Speed: {:.1} tok/s", result.decode_tps);

// Unload model when done (optional, frees memory)
client.unload_model("qwen2.5-0.5b-instruct");
```

## API Reference

### `LocalLLMClient::new(models_dir: Option<PathBuf>) -> Result<Self, String>`

Creates a new `LocalLLMClient` instance.

- `models_dir`: Optional path to models directory. If `None`, uses default `~/.mofa/models/`
- Returns: `Ok(LocalLLMClient)` on success, or `Err(String)` on failure

### `list_models() -> Vec<String>`

Returns a list of all available model IDs (both loaded and discovered).

### `is_model_loaded(model_id: &str) -> bool`

Checks if a model is currently loaded in memory.

### `get_model_info(model_id: &str) -> Option<ModelEntry>`

Gets model entry information for a given model ID.

### `load_model(model_id: &str) -> Result<(), String>`

Loads a model by its ID. The model must be present in the discovered models list.

- Returns: `Ok(())` on success, or `Err(String)` if the model cannot be found or loaded

### `load_model_from_path(model_path: impl AsRef<Path>, model_id: Option<&str>) -> Result<String, String>`

Loads a model from a direct path. Useful when you have a model path that hasn't been discovered yet.

- `model_path`: Path to the model directory
- `model_id`: Optional model ID. If `None`, uses the directory name
- Returns: The model ID on success

### `unload_model(model_id: &str)`

Unloads a model from memory, freeing resources.

### `chat(model_id: &str, messages: &[(String, String)], max_tokens: usize, temperature: f32) -> Result<ChatResult, String>`

Runs chat completion inference.

- `model_id`: The ID of the loaded model
- `messages`: Vector of (role, content) tuples. Roles should be "system", "user", or "assistant"
- `max_tokens`: Maximum number of tokens to generate
- `temperature`: Sampling temperature (0.0 to 1.0+)
- Returns: `Ok(ChatResult)` on success

### `scan_models_dir()`

Scans the models directory for new models and updates the internal registry.

### `models_dir() -> PathBuf`

Gets the current models directory path.

### `config() -> AppConfig`

Gets a copy of the current configuration.

## Thread Safety

`LocalLLMClient` is thread-safe and can be shared across threads using `Arc`:

```rust
use std::sync::Arc;

let client = Arc::new(LocalLLMClient::new(None)?);

// Can be cloned and shared
let client_clone = Arc::clone(&client);
```

## Concurrency and Thread Safety

**Important Concurrency Notes:**

- **Multiple models**: Different models can run inference concurrently
- **Per-model concurrency**: Only one inference can run per model at a time
  - Concurrent requests to the same model will block sequentially
  - This is due to MLX inference requiring mutable access to model state (KV cache, etc.)
  - The write lock ensures thread safety but serializes requests per model

**For async contexts**, use `tokio::task::spawn_blocking()`:

```rust
use std::sync::Arc;
use tokio::task;

let client = Arc::new(LocalLLMClient::new(None)?);
let client_clone = Arc::clone(&client);

// In async function
let result = task::spawn_blocking(move || {
    client_clone.chat("model-id", &messages, 128, 0.7)
}).await??;
```

**Example: Concurrent requests to different models:**

```rust
use std::sync::Arc;
use tokio::task;

let client = Arc::new(LocalLLMClient::new(None)?);

// These will run concurrently (different models)
let handle1 = task::spawn_blocking({
    let c = Arc::clone(&client);
    move || c.chat("model-1", &messages1, 128, 0.7)
});

let handle2 = task::spawn_blocking({
    let c = Arc::clone(&client);
    move || c.chat("model-2", &messages2, 128, 0.7)
});

let (r1, r2) = tokio::join!(handle1, handle2);
```

**Example: Sequential requests to same model:**

```rust
use std::sync::Arc;
use tokio::task;

let client = Arc::new(LocalLLMClient::new(None)?);

// These will run sequentially (same model)
let r1 = task::spawn_blocking({
    let c = Arc::clone(&client);
    move || c.chat("model-1", &messages1, 128, 0.7)
}).await??;

let r2 = task::spawn_blocking({
    let c = Arc::clone(&client);
    move || c.chat("model-1", &messages2, 128, 0.7)  // Same model
}).await??;
```

## Async Usage Example

Here's a complete example showing how to use `LocalLLMClient` in an async context:

```rust
use std::sync::Arc;
use mofa_local_llm::LocalLLMClient;
use tokio::task;

async fn run_inference() -> Result<(), Box<dyn std::error::Error>> {
    // Create client
    let client = Arc::new(LocalLLMClient::new(None)?);
    
    // Load model (can be done in blocking context or async with spawn_blocking)
    let client_clone = Arc::clone(&client);
    task::spawn_blocking(move || {
        client_clone.load_model("qwen2.5-0.5b-instruct")
    }).await??;
    
    // Run inference
    let messages = vec![
        ("system".to_string(), "You are helpful.".to_string()),
        ("user".to_string(), "Hello!".to_string()),
    ];
    
    let client_clone = Arc::clone(&client);
    let result = task::spawn_blocking(move || {
        client_clone.chat("qwen2.5-0.5b-instruct", &messages, 128, 0.7)
    }).await??;
    
    println!("Response: {}", result.text);
    Ok(())
}
```

## Supported Model Types

The library automatically detects the backend from the model's `config.json`:

- `qwen2` → Qwen2 backend
- `qwen3` or `qwen` → Qwen3 backend
- `mistral` → Mistral backend
- `glm4` or `chatglm` → GLM-4 backend
- `mixtral` → Mixtral backend
- `minicpm` or `minicpm4` → MiniCPM-SALA backend

## Error Handling

All methods return `Result<T, String>` where the error string provides a descriptive message about what went wrong.

Common errors:
- Model not found: "Model 'X' not found. Use list_models() to see available models."
- Model not loaded: "Model 'X' is not loaded. Call load_model() first."
- Unknown model type: "Unknown model type: X"
- Load failure: "Failed to load model 'X': <details>"

## Integration with MoFA

This library API is designed to be used by MoFA's `LocalLLMProvider` implementation. The provider will:

1. Create a `LocalLLMClient` instance
2. Load models on demand
3. Convert MoFA's message types to `(String, String)` tuples
4. Convert `ChatResult` back to MoFA's response types
5. Handle errors appropriately

## Backward Compatibility

The existing HTTP server (`mofa-server`) and GUI (`mofa-gui`) binaries continue to work unchanged. The library API is additive and does not break existing functionality.
