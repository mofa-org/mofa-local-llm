use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Html, Json};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

use super::types::*;
use crate::config::AppConfig;
use crate::download;
use crate::inference::{ChatResult, InferenceRequest};

use tokio::sync::{oneshot, RwLock};

/// Shared server state.
pub struct AppState {
    pub inference_tx: tokio::sync::mpsc::Sender<InferenceRequest>,
    pub config: RwLock<AppConfig>,
    pub default_temperature: f32,
    pub default_max_tokens: usize,
    pub loaded_model_id: RwLock<Option<String>>,
}

pub type SharedState = Arc<AppState>;

// ============================================================================
// Health Check
// ============================================================================

pub async fn health_check() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

// ============================================================================
// Chat Completion
// ============================================================================

pub async fn chat_completion(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<ApiError>)> {
    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new("messages array is empty", "invalid_request_error")),
        ));
    }

    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();

    let max_tokens = req.max_tokens.unwrap_or(state.default_max_tokens);
    let temperature = req.temperature.unwrap_or(state.default_temperature);

    let (resp_tx, resp_rx) = oneshot::channel();

    state
        .inference_tx
        .send(InferenceRequest::Chat {
            messages,
            max_tokens,
            temperature,
            response_tx: resp_tx,
        })
        .await
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError::new("Inference worker unavailable", "server_error")),
            )
        })?;

    let result = resp_rx.await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new("Inference channel closed", "server_error")),
        )
    })?;

    match result {
        Ok(chat_result) => {
            eprintln!(
                "[api] Generated {} tokens ({:.0}ms prefill, {:.1} tok/s)",
                chat_result.completion_tokens, chat_result.prefill_ms, chat_result.decode_tps
            );

            Ok(Json(ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion".to_string(),
                model: req.model,
                choices: vec![Choice {
                    index: 0,
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content: chat_result.text,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens: chat_result.prompt_tokens,
                    completion_tokens: chat_result.completion_tokens,
                    total_tokens: chat_result.prompt_tokens + chat_result.completion_tokens,
                },
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new(e, "server_error")),
        )),
    }
}

// ============================================================================
// Model Management
// ============================================================================

pub async fn list_models(
    State(state): State<SharedState>,
) -> Json<ModelListResponse> {
    let mut config = state.config.write().await;
    config.scan_models_dir();

    let loaded_id = state.loaded_model_id.read().await;

    let data: Vec<ModelInfo> = config
        .models
        .iter()
        .map(|m| {
            let is_loaded = loaded_id.as_deref() == Some(&m.id);
            ModelInfo {
                id: m.id.clone(),
                object: "model".to_string(),
                repo_id: if m.repo_id.is_empty() {
                    None
                } else {
                    Some(m.repo_id.clone())
                },
                model_type: m.model_type.clone(),
                path: m.path.clone(),
                loaded: is_loaded,
                size_bytes: m.size_bytes,
                downloaded_at: m.downloaded_at.clone(),
            }
        })
        .collect();

    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

pub async fn download_model(
    State(state): State<SharedState>,
    Json(req): Json<DownloadRequest>,
) -> Result<(StatusCode, Json<Value>), (StatusCode, Json<ApiError>)> {
    if req.repo_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new("repo_id is required", "invalid_request_error")),
        ));
    }

    let model_id = req
        .repo_id
        .split('/')
        .last()
        .unwrap_or(&req.repo_id)
        .to_string();

    // Check if already exists
    {
        let config = state.config.read().await;
        if config.models.iter().any(|m| m.id == model_id) {
            return Err((
                StatusCode::CONFLICT,
                Json(ApiError::new(
                    format!("Model '{}' already exists", model_id),
                    "conflict",
                )),
            ));
        }
    }

    let models_dir = PathBuf::from(&state.config.read().await.models_dir);
    let repo_id = req.repo_id.clone();
    let state_clone = state.clone();

    tokio::task::spawn_blocking(move || {
        eprintln!("[download] Starting: {}", repo_id);
        match download::download_model(&repo_id, &models_dir, None) {
            Ok(entry) => {
                let mut config = state_clone.config.blocking_write();
                download::register_model(&mut config, entry);
                eprintln!("[download] Complete: {}", repo_id);
            }
            Err(e) => {
                eprintln!("[download] Failed: {}: {}", repo_id, e);
                let dest = models_dir.join(repo_id.split('/').last().unwrap_or(&repo_id));
                let _ = std::fs::remove_dir_all(&dest);
            }
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(json!({
            "status": "downloading",
            "id": model_id,
            "repo_id": req.repo_id,
        })),
    ))
}

pub async fn delete_model(
    State(state): State<SharedState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<ApiError>)> {
    let mut config = state.config.write().await;

    let idx = config
        .models
        .iter()
        .position(|m| m.id == model_id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("Model not found: {}", model_id),
                    "not_found",
                )),
            )
        })?;

    // Prevent deleting loaded model
    let loaded_id = state.loaded_model_id.read().await;
    if loaded_id.as_deref() == Some(&model_id) {
        return Err((
            StatusCode::CONFLICT,
            Json(ApiError::new(
                "Cannot delete the currently loaded model",
                "conflict",
            )),
        ));
    }

    let path = PathBuf::from(&config.models[idx].path);
    if path.exists() {
        std::fs::remove_dir_all(&path).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError::new(
                    format!("Failed to remove directory: {}", e),
                    "server_error",
                )),
            )
        })?;
    }

    config.models.remove(idx);
    let _ = config.save();

    Ok(Json(json!({"id": model_id, "deleted": true})))
}

// ============================================================================
// Audio Transcription
// ============================================================================

pub async fn transcribe_audio(
    State(state): State<SharedState>,
    body: axum::body::Bytes,
) -> Result<Json<TranscriptionResponse>, (StatusCode, Json<ApiError>)> {
    let (resp_tx, resp_rx) = oneshot::channel();

    // Parse WAV from body bytes
    let samples = parse_wav_bytes(&body).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(e, "invalid_request_error")),
        )
    })?;

    state
        .inference_tx
        .send(InferenceRequest::Transcribe {
            audio_samples: samples,
            sample_rate: 16000,
            response_tx: resp_tx,
        })
        .await
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError::new("Inference worker unavailable", "server_error")),
            )
        })?;

    let result = resp_rx.await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new("Inference channel closed", "server_error")),
        )
    })?;

    match result {
        Ok(text) => Ok(Json(TranscriptionResponse { text })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new(e, "server_error")),
        )),
    }
}

fn parse_wav_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let cursor = std::io::Cursor::new(bytes);
    let reader = hound::WavReader::new(cursor).map_err(|e| format!("Invalid WAV: {}", e))?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert stereo to mono if needed
    let mono = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if needed
    Ok(crate::inference::asr::resample_to_16khz(&mono, spec.sample_rate))
}

// ============================================================================
// Model Catalog
// ============================================================================

pub async fn list_catalog() -> Json<Value> {
    let catalog: Vec<Value> = crate::models::MODEL_CATALOG
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "name": m.name,
                "category": format!("{}", m.category),
                "engine": m.engine,
                "repo_id": m.repo_id,
                "description": m.description,
                "size_hint": m.size_hint,
            })
        })
        .collect();

    Json(json!({"object": "list", "data": catalog}))
}

// ============================================================================
// Web Chat UI
// ============================================================================

pub async fn index_html(State(state): State<SharedState>) -> Html<String> {
    let model_id = state
        .loaded_model_id
        .read()
        .await
        .clone()
        .unwrap_or_else(|| "none".to_string());
    Html(build_chat_html(&model_id))
}

fn build_chat_html(model_id: &str) -> String {
    format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MOFA Local LLM</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }}
header {{ background: #1a1a2e; padding: 16px 24px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #2a2a3e; }}
header h1 {{ font-size: 20px; color: #7c83ff; }}
header .model-badge {{ background: #2a2a3e; color: #a0a0c0; padding: 4px 12px; border-radius: 12px; font-size: 13px; }}
header .status {{ margin-left: auto; color: #4caf50; font-size: 13px; }}
#chat {{ flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }}
.msg {{ max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
.msg.user {{ align-self: flex-end; background: #2a2a5e; color: #d0d0ff; border-bottom-right-radius: 4px; }}
.msg.assistant {{ align-self: flex-start; background: #1e1e1e; color: #e0e0e0; border-bottom-left-radius: 4px; border: 1px solid #2a2a2a; }}
.msg.assistant pre {{ background: #111; padding: 12px; border-radius: 8px; overflow-x: auto; margin: 8px 0; }}
.msg.assistant code {{ font-family: 'SF Mono', Monaco, Menlo, monospace; font-size: 13px; }}
.msg .meta {{ font-size: 11px; color: #666; margin-top: 6px; }}
#input-area {{ padding: 16px 24px; background: #1a1a1a; border-top: 1px solid #2a2a2a; display: flex; gap: 12px; }}
#user-input {{ flex: 1; background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 12px; padding: 12px 16px; color: #e0e0e0; font-size: 15px; outline: none; resize: none; min-height: 48px; max-height: 120px; font-family: inherit; }}
#user-input:focus {{ border-color: #7c83ff; }}
#send-btn {{ background: #7c83ff; border: none; border-radius: 12px; padding: 0 24px; color: white; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.2s; }}
#send-btn:hover {{ background: #6a70e0; }}
#send-btn:disabled {{ background: #3a3a4a; cursor: not-allowed; }}
.typing {{ display: inline-flex; gap: 4px; padding: 8px 16px; }}
.typing span {{ width: 8px; height: 8px; background: #555; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out; }}
.typing span:nth-child(1) {{ animation-delay: -0.32s; }}
.typing span:nth-child(2) {{ animation-delay: -0.16s; }}
@keyframes bounce {{ 0%, 80%, 100% {{ transform: scale(0); }} 40% {{ transform: scale(1); }} }}
</style>
</head>
<body>
<header>
  <h1>MOFA Local LLM</h1>
  <span class="model-badge">{model_id}</span>
  <span class="status" id="status">Ready</span>
</header>
<div id="chat">
  <div class="msg assistant">Welcome! I'm running locally on your Apple Silicon. Ask me anything.</div>
</div>
<div id="input-area">
  <textarea id="user-input" rows="1" placeholder="Type a message..." autofocus></textarea>
  <button id="send-btn" onclick="sendMessage()">Send</button>
</div>
<script>
const MODEL = "{model_id}";
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const statusEl = document.getElementById('status');
let messages = [];

inputEl.addEventListener('keydown', e => {{
  if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); sendMessage(); }}
}});
inputEl.addEventListener('input', () => {{
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
}});

function addMsg(role, content, meta) {{
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = content;
  if (meta) {{
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    div.appendChild(m);
  }}
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}}

function showTyping() {{
  const div = document.createElement('div');
  div.className = 'msg assistant typing';
  div.id = 'typing';
  div.innerHTML = '<span></span><span></span><span></span>';
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}}
function hideTyping() {{
  const el = document.getElementById('typing');
  if (el) el.remove();
}}

async function sendMessage() {{
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;
  statusEl.textContent = 'Generating...';
  statusEl.style.color = '#ff9800';

  addMsg('user', text);
  messages.push({{ role: 'user', content: text }});

  showTyping();
  const t0 = performance.now();

  try {{
    const resp = await fetch('/v1/chat/completions', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        model: MODEL,
        messages: messages,
        max_tokens: 2048,
        temperature: 0.7,
      }}),
    }});

    hideTyping();
    const data = await resp.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    if (data.choices && data.choices.length > 0) {{
      const reply = data.choices[0].message.content;
      const tokens = data.usage ? data.usage.completion_tokens : '?';
      const tps = data.usage ? (data.usage.completion_tokens / (performance.now() - t0) * 1000).toFixed(1) : '?';
      addMsg('assistant', reply, tokens + ' tokens | ' + tps + ' tok/s | ' + elapsed + 's');
      messages.push({{ role: 'assistant', content: reply }});
    }} else if (data.error) {{
      addMsg('assistant', 'Error: ' + (data.error.message || JSON.stringify(data.error)));
    }}
  }} catch (e) {{
    hideTyping();
    addMsg('assistant', 'Network error: ' + e.message);
  }}

  sendBtn.disabled = false;
  statusEl.textContent = 'Ready';
  statusEl.style.color = '#4caf50';
  inputEl.focus();
}}
</script>
</body>
</html>"##, model_id = model_id)
}

