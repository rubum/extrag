//! # Advanced Retrieval Engine
//! 
//! Orchestrates a multi-stage search process combining hypothetical expansion, 
//! vector similarity, and Reinforcement Learning utility profiles (MemRL).

use extrag_core::embeddings::Embedder;
use extrag_core::error::ExtragError;
use extrag_core::llm::LlmClient;
use extrag_core::reranker::ReRanker;
use extrag_core::vector_store::{SearchFilter, SearchResult, VectorStore};

/// Configuration for the retrieval engine's behavior.
pub struct RetrievalConfig {
    /// Maximum number of documents to return.
    pub top_k: usize,
    /// Whether to generate a Hypothetical Document before searching.
    pub use_hyde: bool,
    /// Weight given to semantic similarity (Cosine/L2) in fusion.
    pub semantic_weight: f32,
    /// Weight given to historical utility ($Q$-score) in fusion.
    pub utility_weight: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            use_hyde: false,
            semantic_weight: 0.7,
            utility_weight: 0.3,
        }
    }
}

/// A sophisticated retrieval orchestrator capable of agentic search patterns.
pub struct AdvancedRetrievalEngine {
    embedder: Box<dyn Embedder>,
    vector_store: Box<dyn VectorStore>,
    llm_client: Option<Box<dyn LlmClient>>,
    reranker: Option<Box<dyn ReRanker>>,
}

impl AdvancedRetrievalEngine {
    /// Creates a new AdvancedRetrievalEngine with mandatory backends.
    pub fn new(embedder: Box<dyn Embedder>, vector_store: Box<dyn VectorStore>) -> Self {
        Self {
            embedder,
            vector_store,
            llm_client: None,
            reranker: None,
        }
    }

    /// Attaches an optional LLM client to support HyDE and Query Expansion.
    pub fn with_llm(mut self, llm: Box<dyn LlmClient>) -> Self {
        self.llm_client = Some(llm);
        self
    }

    /// Attaches an optional Re-Ranker to support cross-encoder scoring.
    pub fn with_reranker(mut self, reranker: Box<dyn ReRanker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Executes the multi-stage, agentic retrieval process.
    /// 
    /// # The Stages:
    /// 1. **HyDE**: Generates a hypothetical expansion if `use_hyde` is enabled.
    /// 2. **Vector Search**: Performs an initial dense retrieval.
    /// 3. **MemRL Fusion**: Balances semantic scores with historical utility using Z-score normalization.
    /// 4. **Optional Re-Ranking**: Final pass through a cross-encoder model.
    ///
    /// # Errors
    /// Returns [`ExtragError`] if any backend client fails during retrieval.
    pub async fn retrieve(
        &self,
        query: &str,
        config: RetrievalConfig,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, ExtragError> {
        // Stage 1: Context Augmentation (HyDE)
        let search_query = if config.use_hyde && self.llm_client.is_some() {
            let llm = self.llm_client.as_ref().unwrap();
            let hypo = llm.generate_hypothetical_document(query).await;
            match hypo {
                Ok(doc) => {
                    tracing::debug!("Generated HyDE document: {}", doc);
                    doc
                }
                Err(e) => {
                    tracing::warn!("HyDE stage failed, falling back to raw query: {}", e);
                    query.to_string()
                }
            }
        } else {
            query.to_string()
        };

        // Stage 2: Embed and Vector Search
        let embedding = self.embedder.embed(&search_query).await?;
        let mut results = self
            .vector_store
            .search(embedding, config.top_k * 2, filter) // fetch oversample for re-ranking
            .await?;

        if results.is_empty() {
            return Ok(results);
        }

        // Stage 3: MemRL Z-Score (Value-Aware) Fusion
        // We normalize utility to [0, 1] across the local result set
        let max_utility = results
            .iter()
            .map(|r| r.document.utility)
            .reduce(f32::max)
            .unwrap_or(0.0);
        let min_utility = results
            .iter()
            .map(|r| r.document.utility)
            .reduce(f32::min)
            .unwrap_or(0.0);

        let utility_range = max_utility - min_utility;

        for r in &mut results {
            let normalized_utility = if utility_range > 0.001 {
                (r.document.utility - min_utility) / utility_range
            } else {
                0.0
            };

            // Calculate the final agentic score
            let fused_score =
                (config.semantic_weight * r.score) + (config.utility_weight * normalized_utility);
            r.score = fused_score;
        }

        // Re-sort results based on the new fused value
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Optional Stage 4: Cross-Encoder Re-ranking
        if let Some(reranker) = &self.reranker {
            results = reranker.rerank(query, results).await?;
        } else {
            // Apply truncation to top_k if no external reranker is used
            results.truncate(config.top_k);
        }

        Ok(results)
    }
}
