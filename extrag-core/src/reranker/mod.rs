//! # Re-Ranking Interfaces
//! 
//! Provides abstractions for the final stage of retrieval where results are 
//! re-evaluated using high-precision Cross-Encoder models.

use crate::error::ExtragError;
use crate::vector_store::SearchResult;
use async_trait::async_trait;

/// Trait for re-ranking search results using more expensive Cross-Encoder models.
/// 
/// Unlike Bi-Encoders used for initial vector search, Cross-Encoders evaluate 
/// the query and document simultaneously to produce a more accurate relevance score.
#[async_trait]
pub trait ReRanker: Send + Sync {
    /// Re-scores and re-orders a list of search results based on the original query.
    /// 
    /// # Arguments
    /// * `query` - The user's original natural language query.
    /// * `results` - The set of results fetched during the initial retrieval phase.
    ///
    /// # Errors
    /// Returns [`ExtragError::LlmError`] if the re-ranking model fails to process the input.
    async fn rerank(
        &self,
        query: &str,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, ExtragError>;
}
