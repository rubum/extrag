//! # Vector Embeddings
//! 
//! Provides abstractions for converting text into high-dimensional numerical vectors.
//! These vectors facilitate semantic search and contextual retrieval.

use crate::error::ExtragError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Represents a numerical vector resulting from an embedding model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Embedding(pub Vec<f32>);

/// Trait for generating high-dimensional embeddings from text strings.
/// 
/// Implementations may use local models (like BERT/Nomic) or remote APIs (OpenAI/Gemini).
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Generates a single embedding for the provided text.
    ///
    /// # Errors
    /// Returns [`ExtragError::EmbeddingError`] if the model fails or connection is lost.
    async fn embed(&self, text: &str) -> Result<Embedding, ExtragError>;

    /// Generates multiple embeddings for a batch of text chunks.
    ///
    /// This is often more efficient than calling `embed` repeatedly due to batching support in models.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>, ExtragError>;

    /// Returns the dimensionality (length) of the vectors produced by this embedder.
    fn dimension(&self) -> usize;
}
