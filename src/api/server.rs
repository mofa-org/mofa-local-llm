use axum::routing::{delete, get, post};
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

use super::handlers::{self, AppState, SharedState};
use crate::config::AppConfig;
use crate::inference::InferenceRequest;

/// Build the API router with all routes.
pub fn build_router(state: SharedState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // Web UI
        .route("/", get(handlers::index_html))
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(handlers::chat_completion))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/download", post(handlers::download_model))
        .route("/v1/models/:model_id", delete(handlers::delete_model))
        .route("/v1/audio/transcriptions", post(handlers::transcribe_audio))
        // Extended endpoints
        .route("/v1/catalog", get(handlers::list_catalog))
        .route("/health", get(handlers::health_check))
        .layer(cors)
        .with_state(state)
}

/// Start the API server.
pub async fn start_server(
    port: u16,
    inference_tx: tokio::sync::mpsc::Sender<InferenceRequest>,
    config: AppConfig,
    loaded_model_id: Option<String>,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        inference_tx,
        config: tokio::sync::RwLock::new(config),
        default_temperature: 0.7,
        default_max_tokens: 2048,
        loaded_model_id: tokio::sync::RwLock::new(loaded_model_id),
    });

    let app = build_router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    eprintln!();
    eprintln!("  MOFA Local LLM Server");
    eprintln!("  http://localhost:{}", port);
    eprintln!();
    eprintln!("  Endpoints:");
    eprintln!("    GET    /                          - Web Chat UI");
    eprintln!("    POST   /v1/chat/completions       - Chat completion (OpenAI API)");
    eprintln!("    GET    /v1/models                 - List downloaded models");
    eprintln!("    POST   /v1/models/download        - Download from HuggingFace");
    eprintln!("    DELETE /v1/models/{{id}}            - Delete a model");
    eprintln!("    POST   /v1/audio/transcriptions   - Transcribe audio (WAV)");
    eprintln!("    GET    /v1/catalog                - List all available models");
    eprintln!("    GET    /health                    - Health check");
    eprintln!();

    axum::serve(listener, app).await?;
    Ok(())
}
