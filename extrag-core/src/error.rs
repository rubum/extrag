//! # Extrag Error System
//! 
//! Centralizes all error types encountered across the Extrag platform,
//! leveraging `thiserror` for idiomatic Rust error handling and reporting.

use thiserror::Error;

/// Core error type used throughout the Extrag platform.
/// 
/// This enum categorizes failure modes across ETL, Embedding, Vector Storage, and LLM inference.
#[derive(Debug, Error)]
pub enum ExtragError {
    /// Issued when a document or metadata payload cannot be parsed or transformed.
    #[error("Failed to parse payload: {0}")]
    ParseError(String),

    /// Issued when an external resource (Database, LLM API, Filesystem) is unreachable.
    #[error("Connection error: {0}")]
    ConnectionError(String),

    /// Missing, malformed, or corrupt data encountered during document processing.
    #[error("Invalid data encountered: {0}")]
    InvalidData(String),

    /// Critical failure during the recursive or token-based splitting process.
    #[error("Chunking error: {0}")]
    ChunkingError(String),

    /// Failure to generate search vectors, either due to model error or API limits.
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    /// Failure during LLM text generation (e.g., HyDE or Query Expansion).
    #[error("LLM error: {0}")]
    LlmError(String),

    /// Error returned by the underlying vector store (e.g., Qdrant or Pinecone).
    #[error("Vector store error: {0}")]
    VectorStoreError(String),
}
