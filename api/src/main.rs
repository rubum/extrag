//! # Extrag: The Agentic ETL-to-RAG Engine
//!
//! This module provides the Axum-based REST API for the Extrag ETL-to-RAG Engine.
//! It exposes endpoints for document ingestion, advanced agentic retrieval (HyDE + Multi-Query),
//! and explicit feedback loops for Reinforcement Learning (Utility scoring).

use axum::{
    Router,
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::services::ServeDir;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use extrag_core::chunker::RecursiveCharacterChunker;
use extrag_core::ollama::OllamaClient;
use extrag_core::qdrant::QdrantVectorStore;
use extrag_core::vector_store::VectorStore;

use rag::ingestion::IngestionPipeline;
use rag::retrieval::{AdvancedRetrievalEngine, RetrievalConfig};
use rag::sync_state::SqliteSyncStateStore;

// --- Engine Configuration Constants ---
const SERVER_PORT: u16 = 8080;
// Fallbacks are now handled in main()
const COLLECTION_NAME: &str = "extrag_knowledge";
const CHUNK_BATCH_SIZE: usize = 10;

/// Shared application state managed by Axum.
#[derive(Clone)]
struct AppState {
    vector_store: Arc<QdrantVectorStore>,
    llm_client: Arc<OllamaClient>,
    sync_store: Arc<SqliteSyncStateStore>,
}

// --- API Payloads ---

#[derive(Deserialize)]
struct IngestRequest {
    /// Absolute or relative path to the directory to ingest.
    path: String,
}

#[derive(Serialize)]
struct IngestResponse {
    message: String,
    chunks_indexed: usize,
}

#[derive(Deserialize)]
struct RetrieveRequest {
    /// The user's query.
    query: String,
    /// Number of top pieces of context to return (default: 5).
    top_k: Option<usize>,
    /// Whether to generate a Hypothetical Document (HyDE) before searching (default: true).
    use_hyde: Option<bool>,
}

#[derive(Serialize)]
struct RetrieveResponse {
    /// The returned context chunks containing the data and utility profiles.
    results: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct FeedbackRequest {
    /// The unique Vector ID of the chunk to update.
    document_id: String,
    /// The RL reward to apply (e.g., +1.0 for helpful, -1.0 for unhelpful).
    reward: f32,
}

#[derive(Serialize)]
struct FeedbackResponse {
    message: String,
}

// --- Main Application Entrypoint ---

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "info,extrag_core=debug,rag=debug,api=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Initializing Extrag Engine...");

    let ollama_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let ollama_model =
        std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:latest".to_string());
    let qdrant_url =
        std::env::var("QDRANT_BASE_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());

    // Initialize backend connectors
    let embedder = OllamaClient::new(&ollama_url, Some(ollama_model));
    let vector_store = QdrantVectorStore::new(&qdrant_url, COLLECTION_NAME);
    let sqlite_path =
        std::env::var("SQLITE_DB_PATH").unwrap_or_else(|_| "extrag_state.db".to_string());
    let sync_store = SqliteSyncStateStore::new(&sqlite_path).await?;

    let state = AppState {
        vector_store: Arc::new(vector_store),
        llm_client: Arc::new(embedder),
        sync_store: Arc::new(sync_store),
    };

    let web_dir = std::env::var("WEB_DIR").unwrap_or_else(|_| "../web".to_string());

    // Configure router
    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/v1/ingest", post(handle_ingest))
        .route("/v1/retrieve", post(handle_retrieve))
        .route("/v1/feedback", post(handle_feedback))
        .fallback_service(ServeDir::new(web_dir))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], SERVER_PORT));
    tracing::info!("Extrag API listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// --- Route Handlers ---

/// `POST /v1/ingest`
/// Initiates a delta-aware extraction and chunking process on a filesystem directory.
async fn handle_ingest(
    State(state): State<AppState>,
    Json(payload): Json<IngestRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    tracing::info!("Starting ingestion for path: {}", payload.path);

    let embedder = Box::new((*state.llm_client).clone());
    let vector_store = Box::new((*state.vector_store).clone());
    let sync_store = Box::new((*state.sync_store).clone());
    let chunker = Box::new(RecursiveCharacterChunker::default());

    let path = payload.path.trim();
    let extractors: Vec<Box<dyn extrag_core::etl::BatchExtractor>> =
        vec![Box::new(etl::FilesystemExtractor::new(path))];

    let parsers: Vec<Box<dyn extrag_core::etl::Parser>> = vec![
        Box::new(etl::MarkdownParser),
        Box::new(etl::JsonParser),
        Box::new(etl::PlainTextParser),
    ];

    let pipeline = IngestionPipeline {
        extractors,
        parsers,
        chunker,
        embedder,
        vector_store,
        sync_state: Some(sync_store),
        chunk_batch_size: CHUNK_BATCH_SIZE,
    };

    let chunks_indexed = pipeline.run().await.map_err(|e| {
        tracing::error!("Ingestion Pipeline execution failed: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Ingestion failed: {}", e),
        )
    })?;

    Ok(Json(IngestResponse {
        message: "Ingestion completed successfully".to_string(),
        chunks_indexed,
    }))
}

/// `POST /v1/retrieve`
/// Executes an advanced context retrieval utilizing HyDE and MemRL Value-based fusion.
async fn handle_retrieve(
    State(state): State<AppState>,
    Json(payload): Json<RetrieveRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    tracing::info!("Retrieving context for query: {}", payload.query);

    let embedder = Box::new((*state.llm_client).clone());
    let vector_store = Box::new((*state.vector_store).clone());
    let llm_client = Box::new((*state.llm_client).clone());

    let engine = AdvancedRetrievalEngine::new(embedder, vector_store).with_llm(llm_client);

    let config = RetrievalConfig {
        top_k: payload.top_k.unwrap_or(5),
        use_hyde: payload.use_hyde.unwrap_or(true),
        semantic_weight: 0.7,
        utility_weight: 0.3,
    };

    let results = engine
        .retrieve(&payload.query, config, None)
        .await
        .map_err(|e| {
            tracing::error!("Retrieval Engine failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Retrieval failed: {}", e),
            )
        })?;

    let json_results: Vec<serde_json::Value> = results
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "score": r.score,
                "id": r.document.id,
                "utility": r.document.utility,
                "content": r.document.chunk.content,
                "source": r.document.chunk.source_id,
            })
        })
        .collect();

    Ok(Json(RetrieveResponse {
        results: json_results,
    }))
}

/// `POST /v1/feedback`
/// Assigns a Reinforcement Learning reward (Delta) to the specific chunk's Utility Q-score.
async fn handle_feedback(
    State(state): State<AppState>,
    Json(payload): Json<FeedbackRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    tracing::info!(
        "Applying feedback reward {} to document {}",
        payload.reward,
        payload.document_id
    );

    state
        .vector_store
        .update_utility(&payload.document_id, payload.reward)
        .await
        .map_err(|e| {
            tracing::error!("Utility update failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to update utility: {}", e),
            )
        })?;

    Ok(Json(FeedbackResponse {
        message: "Utility profile updated successfully".to_string(),
    }))
}
