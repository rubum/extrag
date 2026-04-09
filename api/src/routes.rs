//! ## API Route Handlers and Payloads
//!
//! This module contains all the core logic for processing API requests. It defines:
//! - **Request/Response Payloads**: Strongly-typed serde structures for all endpoints.
//! - **Ingestion Handlers**: Orchestrating the `IngestionPipeline` for scanning and indexing.
//! - **Retrieval Handlers**: Executing the `AdvancedRetrievalEngine` with agentic features (HyDE).
//! - **Feedback Handlers**: Applying RL-inspired utility rewards to indexed chunks.
//! - **Management Handlers**: Listing and deleting vector collections and managing ingestion cache.
//!
//! Handlers are designed to be stateless, pulling all necessary backend connectors
//! from the shared `AppState`.

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};

use extrag_core::chunker::RecursiveCharacterChunker;
use extrag_core::vector_store::VectorStore;

use rag::ingestion::IngestionPipeline;
use rag::retrieval::{AdvancedRetrievalEngine, RetrievalConfig};
use rag::sync_state::SyncStateStore;

use crate::{AppState, CHUNK_BATCH_SIZE};

// --- API Payloads ---

#[derive(Deserialize)]
pub struct IngestRequest {
    /// Absolute or relative path to the directory to ingest.
    pub path: String,
}

#[derive(Serialize)]
pub struct IngestResponse {
    pub message: String,
    pub chunks_indexed: usize,
}

#[derive(Deserialize)]
pub struct RetrieveRequest {
    /// The user's query.
    pub query: String,
    /// Number of top pieces of context to return (default: 5).
    pub top_k: Option<usize>,
    /// Whether to generate a Hypothetical Document (HyDE) before searching (default: true).
    pub use_hyde: Option<bool>,
}

#[derive(Serialize)]
pub struct RetrieveResponse {
    /// The returned context chunks containing the data and utility profiles.
    pub results: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct FeedbackRequest {
    /// The unique Vector ID of the chunk to update.
    pub document_id: String,
    /// The RL reward to apply (e.g., +1.0 for helpful, -1.0 for unhelpful).
    pub reward: f32,
}

#[derive(Serialize)]
pub struct FeedbackResponse {
    pub message: String,
}

#[derive(Serialize)]
pub struct ListCollectionsResponse {
    pub collections: Vec<String>,
}

#[derive(Serialize)]
pub struct DeleteCollectionResponse {
    pub message: String,
}

// --- Route Handlers ---

/// `POST /v1/ingest`
/// Initiates a delta-aware extraction and chunking process on a filesystem directory.
pub async fn handle_ingest(
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
pub async fn handle_retrieve(
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
pub async fn handle_feedback(
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

/// `GET /v1/collections`
/// Returns a list of all existing collections in the vector store.
pub async fn handle_list_collections(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let collections = state.vector_store.list_collections().await.map_err(|e| {
        tracing::error!("Failed to list collections: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to list collections: {}", e),
        )
    })?;

    Ok(Json(ListCollectionsResponse { collections }))
}

/// `DELETE /v1/collections/{name}`
/// Deletes the specified collection and clears the associated sync state.
pub async fn handle_delete_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    tracing::info!("Deleting collection: {}", name);

    // 1. Delete the collection from the vector store
    state
        .vector_store
        .delete_collection(&name)
        .await
        .map_err(|e| {
            tracing::error!("Failed to delete collection {}: {}", name, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to delete collection: {}", e),
            )
        })?;

    // 2. Wipe the sync state to ensure re-ingestion works correctly
    state.sync_store.clear_all().await.map_err(|e| {
        tracing::error!("Failed to clear sync state: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to clear sync state: {}", e),
        )
    })?;

    Ok(Json(DeleteCollectionResponse {
        message: format!(
            "Collection '{}' and associated sync state cleared successfully.",
            name
        ),
    }))
}

/// `POST /v1/cache/clear`
/// Wipes the local synchronization state, forcing a full re-ingestion of all documents.
pub async fn handle_clear_cache(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    tracing::info!("Manually clearing ingestion cache...");

    state.sync_store.clear_all().await.map_err(|e| {
        tracing::error!("Failed to clear sync state: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to clear sync state: {}", e),
        )
    })?;

    Ok(Json(serde_json::json!({
        "message": "Ingestion cache cleared successfully. Next ingestion will be a full sync."
    })))
}
