//! # Vector Storage Interfaces
//! 
//! Defines the core abstractions for storing and searching high-dimensional vectors.
//! This module standardizes how chunks, embeddings, and utility scores (MemRL) 
//! are handled across different database backends.

use crate::chunker::Chunk;
use crate::embeddings::Embedding;
use crate::error::ExtragError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A stored document chunk representing the content, its metadata, and its embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    /// Unique identifier for this specific chunk (often a UUID or hash).
    pub id: String,

    /// The associated [`Chunk`] content and metadata.
    pub chunk: Chunk,

    /// The numerical [`Embedding`] representing this chunk.
    pub embedding: Embedding,

    /// The MemRL Q-score (historical utility). 
    /// High scores indicate the chunk has been historically useful for retrieval.
    pub utility: f32,
}

/// Filter criteria for refining vector search results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilter {
    /// Optional: Filter results to only include chunks from a specific source ID.
    pub source_id: Option<String>,

    /// Optional: Filter results by a matching metadata key and value.
    pub metadata: Option<(String, String)>,
}

/// The result returned from a similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The retrieved document chunk.
    pub document: VectorDocument,

    /// The similarity score calculated by the vector store (e.g., Cosine or L2).
    pub score: f32,
}

/// A trait for interacting with vector databases (e.g., Qdrant, Milvus, Pinecone).
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Indexes a batch of [`VectorDocument`] items.
    ///
    /// # Errors
    /// Returns [`ExtragError::VectorStoreError`] if indexing fails.
    async fn index(&self, documents: Vec<VectorDocument>) -> Result<(), ExtragError>;

    /// Searches for documents similar to the query embedding.
    ///
    /// # Arguments
    /// * `query` - The embedding vector to search for.
    /// * `top_k` - Maximum number of results to return.
    /// * `filter` - Optional criteria to narrow down the search space.
    async fn search(
        &self,
        query: Embedding,
        top_k: usize,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, ExtragError>;

    /// Updates the MemRL Utility score (Q-score) for a specific document chunk.
    ///
    /// This is used to apply Reinforcement Learning rewards or penalties.
    ///
    /// # Arguments
    /// * `id` - The unique identifier of the chunk.
    /// * `delta` - The amount to add (reward) or subtract (penalty) from the score.
    async fn update_utility(&self, id: &str, delta: f32) -> Result<(), ExtragError>;

    /// Removes all document chunks associated with a specific source document.
    /// 
    /// Useful for updating or re-indexing a modified file.
    async fn delete_by_source_id(&self, source_id: &str) -> Result<(), ExtragError>;
}
