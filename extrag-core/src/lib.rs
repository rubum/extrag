//! # Extrag Core
//! 
//! Provides the foundational abstractions, types, and error handling
//! for the Extrag RAG and ETL platform. This crate standardizes the interfaces
//! for embedding generation, vector storage, LLM inference, and data extraction.
//! 
//! ## Architecture
//! - **Traits**: Highly decoupled architectures (`Chunker`, `VectorStore`, `Embedder`, `LlmClient`, `ReRanker`).
//! - **Types**: Standard communication objects (`VectorDocument`, `SearchResult`, `Chunk`).

pub mod chunker;
pub mod embeddings;
pub mod error;
pub mod etl;
pub mod llm;
pub mod ollama;
pub mod payload;
pub mod qdrant;
pub mod reranker;
pub mod test_utils;
pub mod vector_store;
mod memrl_tests;

// Export foundational errors
pub use error::ExtragError;

// Export payload and chunking mechanics
pub use payload::{Format, RawPayload};
pub use chunker::{Chunk, Chunker, RecursiveCharacterChunker, TokenChunker};

// Export traits for AI inference
pub use embeddings::{Embedder, Embedding};
pub use llm::{LlmClient, PromptTemplate};
pub use reranker::ReRanker;

// Export traits for vector databases
pub use vector_store::{SearchFilter, SearchResult, VectorDocument, VectorStore};

// Export concrete implementations
pub use ollama::OllamaClient;
pub use qdrant::QdrantVectorStore;

// Export ETL traits
pub use etl::{BatchExtractor, Parser, StreamReceiver};
