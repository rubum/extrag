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

use async_stream::stream;
use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{
        IntoResponse,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

use extrag_core::chunker::RecursiveCharacterChunker;
use extrag_core::llm::LlmClient;
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
    /// The synthesized Generative response from the LLM based on extracted chunk context.
    pub generation: String,
    /// The hypothetical document generated (if HyDE was enabled).
    pub hyde_doc: Option<String>,
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

    let embedder = Box::new((*state.embedder).clone());
    let vector_store = Box::new((*state.vector_store).clone());
    let sync_store = Box::new((*state.sync_store).clone());
    let chunker = Box::new(RecursiveCharacterChunker::default());

    let path = payload.path.trim();
    let extractors: Vec<Box<dyn extrag_core::etl::BatchExtractor>> =
        vec![Box::new(etl::FilesystemExtractor::new(path))];

    let parsers: Vec<Box<dyn extrag_core::etl::Parser>> = vec![
        Box::new(etl::MarkdownParser),
        Box::new(etl::JsonParser),
        Box::new(etl::PdfParser),
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
) -> impl IntoResponse {
    let stream = stream! {
        yield Ok::<Event, Infallible>(Event::default().json_data(serde_json::json!({
            "type": "log",
            "message": format!("[ENGINE] Query received: '{}'. Initiating retrieval pipeline...", payload.query)
        })).unwrap());

        let embedder = Box::new((*state.embedder).clone());
        let vector_store = Box::new((*state.vector_store).clone());
        let llm_client = Box::new((*state.llm_client).clone());

        let engine = AdvancedRetrievalEngine::new(embedder, vector_store).with_llm(llm_client);

        let config = RetrievalConfig {
            top_k: payload.top_k.unwrap_or(5),
            use_hyde: payload.use_hyde.unwrap_or(true),
            semantic_weight: 0.7,
            utility_weight: 0.3,
        };

        if config.use_hyde {
            yield Ok::<Event, Infallible>(Event::default().json_data(serde_json::json!({
                "type": "log",
                "message": "[HYDE] Generating hypothetical thought trace..."
            })).unwrap());
        }

        let output = match engine.retrieve(&payload.query, config, None).await {
            Ok(o) => o,
            Err(e) => {
                yield Ok::<Event, Infallible>(Event::default().json_data(serde_json::json!({
                    "type": "error",
                    "message": format!("Retrieval failed: {}", e)
                })).unwrap());
                return;
            }
        };

        yield Ok::<Event, Infallible>(Event::default().json_data(serde_json::json!({
            "type": "log",
            "message": format!("[SEARCH] Found {} contexts. Merging semantic similarity and RL utility...", output.results.len())
        })).unwrap());

        let results = output.results;
        let hyde_doc = output.hyde_doc;

        let context_blocks: Vec<String> = results
            .iter()
            .map(|r| {
                format!(
                    "Source ({}):\n{}",
                    r.document.chunk.source_id, r.document.chunk.content
                )
            })
            .collect();
        let mut combined_context = context_blocks.join("\n\n---\n\n");

        const CONTEXT_BUDGET: usize = 6000;
        if combined_context.len() > CONTEXT_BUDGET {
            yield Ok::<Event, Infallible>(Event::default().json_data(serde_json::json!({
                "type": "log",
                "message": format!("[BUDGET] Context exceeds {} chars. Truncating for synthesis...", CONTEXT_BUDGET)
            })).unwrap());
            combined_context.truncate(CONTEXT_BUDGET);
            combined_context.push_str("\n\n[... Context truncated for brevity ...]");
        }

        let system_prompt = format!(
            "You are Extrag Agent, an expert AI assistant answering a user's question based strictly on the provided context chunks.\n\nCONTEXT:\n{}\n\nAnswer the user's question using only this context. If the answer is not in the context, say so gracefully. Use markdown to format your answer effortlessly with bolded text, lists, and code blocks if applicable.",
            combined_context
        );

        yield Ok::<Event, Infallible>(Event::default().json_data(serde_json::json!({
            "type": "log",
            "message": "[SYNTHESIS] Projecting agentic response via LLM..."
        })).unwrap());

        let generation = match state
            .llm_client
            .generate_with_system(&system_prompt, &payload.query)
            .await {
                Ok(g) => g,
                Err(e) => {
                    yield Ok::<Event, Infallible>(Event::default().json_data(serde_json::json!({
                        "type": "error",
                        "message": format!("Synthesis failed: {}", e)
                    })).unwrap());
                    return;
                }
            };

        yield Ok::<Event, Infallible>(Event::default().json_data(serde_json::json!({
            "type": "log",
            "message": "[FINISH] Pipeline complete. Handing over to UI."
        })).unwrap());

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

        yield Ok::<Event, Infallible>(Event::default().json_data(serde_json::json!({
            "type": "result",
            "data": RetrieveResponse {
                generation,
                hyde_doc,
                results: json_results,
            }
        })).unwrap());
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
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

/// `GET /v1/telemetry`
/// Returns real-time metrics tracked by the telemetry layer.
pub async fn handle_telemetry(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let metrics = &state.metrics;
    Ok(Json(serde_json::json!({
        "total_prompt": metrics.total_prompt.load(std::sync::atomic::Ordering::Relaxed),
        "total_completion": metrics.total_completion.load(std::sync::atomic::Ordering::Relaxed),
        "total_chunks": metrics.total_chunks.load(std::sync::atomic::Ordering::Relaxed),
        "total_bytes": metrics.total_bytes.load(std::sync::atomic::Ordering::Relaxed),
        "total_retrievals": metrics.total_retrievals.load(std::sync::atomic::Ordering::Relaxed),
        "total_errors": metrics.total_errors.load(std::sync::atomic::Ordering::Relaxed),
    })))
}
