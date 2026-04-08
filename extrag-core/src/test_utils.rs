use crate::embeddings::{Embedder, Embedding};
use crate::error::ExtragError;
use crate::vector_store::{SearchResult, VectorDocument, VectorStore};
use async_trait::async_trait;
use std::sync::RwLock;

/// A mock embedder for testing that generates pseudo-embeddings from string lengths.
pub struct MockEmbedder {
    pub dimension: usize,
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Embedding, ExtragError> {
        let mut vec = vec![0.0; self.dimension];
        if self.dimension > 0 {
            // Predictable embedding based on text length for testing
            vec[0] = text.len() as f32;
        }
        Ok(Embedding(vec))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>, ExtragError> {
        let mut results = Vec::new();
        for t in texts {
            results.push(self.embed(t).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// A simple, thread-safe in-memory vector store for testing RAG pipelines.
pub struct InMemoryVectorStore {
    documents: RwLock<Vec<VectorDocument>>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            documents: RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn index(&self, documents: Vec<VectorDocument>) -> Result<(), ExtragError> {
        let mut guard = self
            .documents
            .write()
            .map_err(|_| ExtragError::VectorStoreError("Lock poisoned".into()))?;
        guard.extend(documents);
        Ok(())
    }

    async fn search(
        &self,
        query: Embedding,
        top_k: usize,
        _filter: Option<crate::vector_store::SearchFilter>,
    ) -> Result<Vec<SearchResult>, ExtragError> {
        let guard = self
            .documents
            .read()
            .map_err(|_| ExtragError::VectorStoreError("Lock poisoned".into()))?;

        let mut results: Vec<SearchResult> = guard
            .iter()
            .map(|doc| {
                // Simple L2 distance mock score (inverted so closer = higher score up to 1.0)
                let mut dist_sq = 0.0;
                for (v1, v2) in query.0.iter().zip(doc.embedding.0.iter()) {
                    dist_sq += (v1 - v2).powi(2);
                }
                let score = 1.0 / (1.0 + dist_sq.sqrt());

                SearchResult {
                    document: doc.clone(),
                    score,
                }
            })
            .collect();

        // Sort descending by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(top_k);
        Ok(results)
    }

    async fn update_utility(&self, id: &str, delta: f32) -> Result<(), ExtragError> {
        let mut guard = self
            .documents
            .write()
            .map_err(|_| ExtragError::VectorStoreError("Lock poisoned".into()))?;
        if let Some(doc) = guard.iter_mut().find(|d| d.id == id) {
            doc.utility += delta;
        }
        Ok(())
    }

    async fn delete_by_source_id(&self, source_id: &str) -> Result<(), ExtragError> {
        let mut guard = self
            .documents
            .write()
            .map_err(|_| ExtragError::VectorStoreError("Lock poisoned".into()))?;
        guard.retain(|doc| doc.chunk.source_id != source_id);
        Ok(())
    }
}
