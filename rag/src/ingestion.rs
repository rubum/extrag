//! # Ingestion Pipeline
//! 
//! Provides the primary orchestrator for moving data from raw sources to indexed 
//! vector storage. The pipeline handles deduplication, parsing, chunking, and 
//! batch embedding generation.

use crate::sync_state::{DocumentSyncState, SyncStateStore};
use extrag_core::chunker::{Chunk, Chunker};
use extrag_core::embeddings::Embedder;
use extrag_core::error::ExtragError;
use extrag_core::etl::{BatchExtractor, Parser};
use extrag_core::vector_store::{VectorDocument, VectorStore};
use futures::stream::{self, StreamExt};
use xxhash_rust::xxh64::xxh64;

/// The primary orchestrator for the Extrag data ingestion workflow.
/// 
/// The pipeline executes a multi-stage process:
/// 1. **Extraction**: Fetching raw data from configured extractors.
/// 2. **Delta Check**: Skipping documents that haven't changed since the last sync.
/// 3. **Parsing**: Transforming binary content into UTF-8 text.
/// 4. **Chunking**: Splitting text into semantic or sized units.
/// 5. **Embedding**: Generating high-dimensional vectors in batches.
/// 6. **Indexing**: Inserting results into the vector store.
pub struct IngestionPipeline<'a> {
    /// Active data extractors (e.g., Filesystem, S3).
    pub extractors: Vec<Box<dyn BatchExtractor + 'a>>,
    /// Available parsers for different file formats.
    pub parsers: Vec<Box<dyn Parser + 'a>>,
    /// The logic used to split text into chunks.
    pub chunker: Box<dyn Chunker + 'a>,
    /// The model used to generate embeddings.
    pub embedder: Box<dyn Embedder + 'a>,
    /// The database for storing vectors and metadata.
    pub vector_store: Box<dyn VectorStore + 'a>,
    /// Optional persisted state store for delta extraction tracking.
    pub sync_state: Option<Box<dyn SyncStateStore + 'a>>,
    /// Number of chunks to process in a single embedding/upsert batch.
    pub chunk_batch_size: usize,
}

impl<'a> IngestionPipeline<'a> {
    /// Executes the full ingestion process across all registered extractors.
    /// 
    /// Returns the total number of chunks successfully indexed.
    ///
    /// # Errors
    /// Returns [`ExtragError`] if any core stage (like embedding or indexing) fails fatally.
    #[tracing::instrument(skip(self))]
    pub async fn run(&self) -> Result<usize, ExtragError> {
        let mut ingested_count = 0;
        let mut global_chunks_buffer: Vec<Chunk> = Vec::new();

        for extractor in &self.extractors {
            // Fetch the raw data batch from the source
            let payloads = extractor.fetch_batch(None).await?;
            tracing::info!("Extracted {} payloads from source", payloads.len());

            let mut chunk_stream = stream::iter(payloads)
                .map(|payload| async move {
                    // Stage 2: Delta Analysis
                    let mut needs_update = true;
                    let content_hash = format!("{:x}", xxh64(&payload.content, 0));

                    let last_modified = payload
                        .metadata
                        .get("last_modified")
                        .and_then(|s| s.parse::<i64>().ok());

                    if let Some(ref state_store) = self.sync_state {
                        let state = state_store
                            .get_document_state(&payload.source_id)
                            .await
                            .ok()
                            .flatten();

                        if let Some(state) = state {
                            let timestamp_match =
                                last_modified.is_some() && last_modified == state.last_modified;
                            let hash_match = state.content_hash.as_deref() == Some(&content_hash);

                            if timestamp_match && hash_match {
                                tracing::debug!(
                                    "Skipping unchanged document: {}",
                                    payload.source_id
                                );
                                needs_update = false;
                            }
                        }
                    }

                    if !needs_update {
                        return None;
                    }

                    // Stage 3 & 4: Parsing and Chunking
                    // If we need to update, we first clear the old associated vectors
                    let _ = self
                        .vector_store
                        .delete_by_source_id(&payload.source_id)
                        .await;

                    let parser = self.parsers.iter().find(|p| p.supports(&payload));
                    if let Some(p) = parser {
                        match p.parse(&payload) {
                            Ok(text) => {
                                if !text.trim().is_empty() {
                                    match self.chunker.chunk(&payload.source_id, &text) {
                                        Ok(chunks) => {
                                            // Stage 5: Update sync state with new hash/timestamp
                                            if let Some(ref state_store) = self.sync_state {
                                                let _ = state_store
                                                    .update_document_state(DocumentSyncState {
                                                        source_id: payload.source_id.clone(),
                                                        last_modified,
                                                        content_hash: Some(content_hash),
                                                    })
                                                    .await;
                                            }
                                            return Some(chunks);
                                        }
                                        Err(e) => tracing::error!(
                                            "Failed to chunk source {}: {}",
                                            payload.source_id,
                                            e
                                        ),
                                    }
                                }
                            }
                            Err(e) => tracing::error!(
                                "Failed to parse source {}: {}",
                                payload.source_id,
                                e
                            ),
                        }
                    } else {
                        tracing::warn!(
                            "No parser found for format {:?} in source {}",
                            payload.format,
                            payload.source_id
                        );
                    }
                    None
                })
                .buffer_unordered(10); // Concurrent parsing and chunking

            // Collect chunks from the stream and batch them for embedding
            while let Some(chunk_opt) = chunk_stream.next().await {
                if let Some(chunks) = chunk_opt {
                    global_chunks_buffer.extend(chunks);

                    while global_chunks_buffer.len() >= self.chunk_batch_size {
                        let batch: Vec<Chunk> = global_chunks_buffer
                            .drain(..self.chunk_batch_size)
                            .collect();
                        ingested_count += self.process_chunk_batch(batch).await?;
                    }
                }
            }
        }

        // Flush remaining chunks across all extractors
        if !global_chunks_buffer.is_empty() {
            ingested_count += self.process_chunk_batch(global_chunks_buffer).await?;
        }

        tracing::info!(
            "Ingestion pipeline complete. Total chunks indexed: {}",
            ingested_count
        );
        Ok(ingested_count)
    }

    /// Internal logic for batching embeddings and indexing them into the vector store.
    async fn process_chunk_batch(&self, chunks: Vec<Chunk>) -> Result<usize, ExtragError> {
        if chunks.is_empty() {
            return Ok(0);
        }

        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = self.embedder.embed_batch(&texts).await?;

        // Transform results into VectorDocument objects
        let documents: Vec<VectorDocument> = chunks
            .into_iter()
            .zip(embeddings.into_iter())
            .map(|(chunk, embedding)| VectorDocument {
                chunk,
                embedding,
                id: uuid::Uuid::new_v4().to_string(), // Assign unique ID for persistent tracking
                utility: 0.0,                        // Initialize with neutral MemRL utility
            })
            .collect();

        let count = documents.len();
        self.vector_store.index(documents).await?;
        tracing::debug!("Successfully indexed batch of {} chunks", count);
        Ok(count)
    }
}
