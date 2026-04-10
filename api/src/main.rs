//! # Extrag: The Agentic ETL-to-RAG Engine
//!
//! This is the entry point for the Extrag API server. It is responsible for:
//! - Initializing the tracing subscriber for observability.
//! - Loading environment configuration (Ollama, Qdrant, SQLite).
//! - Initializing shared application state (`AppState`).
//! - Configuring the Axum router and middleware (logging, static file serving).
//! - Binding and serving the web application.
//!
//! The API acts as the bridge between the frontend dashboard and the underlying
//! ETL and RAG logic defined in the `extrag-core`, `etl`, and `rag` crates.

use axum::{
    Router,
    routing::{delete, get, post},
};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::services::ServeDir;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use extrag_core::ollama::OllamaClient;
use extrag_core::qdrant::QdrantVectorStore;

use rag::sync_state::SqliteSyncStateStore;

// --- Engine Configuration Constants ---
pub const SERVER_PORT: u16 = 8080;
// Fallbacks are now handled in main()
pub const COLLECTION_NAME: &str = "extrag_knowledge";
pub const CHUNK_BATCH_SIZE: usize = 10;

/// Shared application state managed by Axum.
#[derive(Clone)]
pub struct AppState {
    pub vector_store: Arc<QdrantVectorStore>,
    pub embedder: Arc<OllamaClient>,
    pub llm_client: Arc<OllamaClient>,
    pub sync_store: Arc<SqliteSyncStateStore>,
    pub metrics: DashboardMetrics,
}

mod routes;
mod telemetry;

use telemetry::{DashboardMetrics, DashboardTelemetryLayer};

// Payloads moved to routes.rs

// --- Main Application Entrypoint ---

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let metrics = DashboardMetrics::default();

    // Initialize structured tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| {
                "info,extrag_core=debug,rag=debug,etl=debug,api=debug,extrag::telemetry=info".into()
            }),
        ))
        .with(tracing_subscriber::fmt::layer())
        .with(DashboardTelemetryLayer::new(metrics.clone()))
        .init();

    tracing::info!("Initializing Extrag Engine...");

    let ollama_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());

    // Load separate models for generation and embedding
    let ollama_embed_model = std::env::var("OLLAMA_EMBED_MODEL")
        .or_else(|_| std::env::var("OLLAMA_MODEL"))
        .unwrap_or_else(|_| "nomic-embed-text:latest".to_string());

    let ollama_llm_model = std::env::var("OLLAMA_LLM_MODEL")
        .or_else(|_| std::env::var("OLLAMA_MODEL"))
        .unwrap_or_else(|_| "gemma4:latest".to_string());

    let qdrant_url =
        std::env::var("QDRANT_BASE_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());

    // Initialize backend connectors
    let embedder = OllamaClient::new(&ollama_url, Some(ollama_embed_model));
    let llm_client = OllamaClient::new(&ollama_url, Some(ollama_llm_model));

    let vector_store = QdrantVectorStore::new(&qdrant_url, COLLECTION_NAME);
    let sqlite_path =
        std::env::var("SQLITE_DB_PATH").unwrap_or_else(|_| "extrag_state.db".to_string());
    let sync_store = SqliteSyncStateStore::new(&sqlite_path).await?;

    let state = AppState {
        vector_store: Arc::new(vector_store),
        embedder: Arc::new(embedder),
        llm_client: Arc::new(llm_client),
        sync_store: Arc::new(sync_store),
        metrics,
    };

    let web_dir = std::env::var("WEB_DIR").unwrap_or_else(|_| "../web".to_string());

    // Configure router
    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/v1/ingest", post(routes::handle_ingest))
        .route("/v1/retrieve", post(routes::handle_retrieve))
        .route("/v1/feedback", post(routes::handle_feedback))
        .route("/v1/collections", get(routes::handle_list_collections))
        .route(
            "/v1/collections/{name}",
            delete(routes::handle_delete_collection),
        )
        .route("/v1/cache/clear", post(routes::handle_clear_cache))
        .route("/v1/telemetry", get(routes::handle_telemetry))
        .fallback_service(ServeDir::new(web_dir))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], SERVER_PORT));
    tracing::info!("Extrag API listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
