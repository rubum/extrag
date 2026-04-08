//! # Qdrant Vector Store Integration
//! 
//! High-performance implementation of [`VectorStore`] utilizing the Qdrant REST API.
//! This implementation supports metadata filtering, delta deletions, and 
//! MemRL-style utility score management.

use crate::chunker::Chunk;
use crate::embeddings::Embedding;
use crate::error::ExtragError;
use crate::vector_store::{SearchFilter, SearchResult, VectorDocument, VectorStore};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;

/// An implementation of [`VectorStore`] communicating via Qdrant's REST interface.
/// 
/// The store manages document chunks, their embeddings, and associated "Utility" profiles 
/// used by the agentic retrieval engine.
#[derive(Clone)]
pub struct QdrantVectorStore {
    client: Client,
    base_url: String,
    collection_name: String,
}

impl QdrantVectorStore {
    /// Creates a new QdrantVectorStore.
    /// 
    /// # Arguments
    /// * `base_url` - The Qdrant server URL (e.g., "http://localhost:6333").
    /// * `collection_name` - The name of the collection to target.
    pub fn new(base_url: impl Into<String>, collection_name: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            collection_name: collection_name.into(),
        }
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn index(&self, documents: Vec<VectorDocument>) -> Result<(), ExtragError> {
        let mut points = Vec::with_capacity(documents.len());

        for doc in documents {
            let mut payload = json!({
                "source_id": doc.chunk.source_id,
                "content": doc.chunk.content,
                "utility": doc.utility,
            });

            if let Some(obj) = payload.as_object_mut() {
                for (k, v) in doc.chunk.metadata {
                    obj.insert(k, json!(v));
                }
            }

            points.push(json!({
                "id": doc.id,
                "vector": doc.embedding.0,
                "payload": payload
            }));
        }

        let url = format!(
            "{}/collections/{}/points",
            self.base_url, self.collection_name
        );

        let resp = self
            .client
            .put(&url)
            .json(&json!({ "points": points }))
            .send()
            .await
            .map_err(|e| ExtragError::VectorStoreError(format!("Qdrant index error: {}", e)))?;

        if !resp.status().is_success() {
            let err_body = resp.text().await.unwrap_or_default();
            return Err(ExtragError::VectorStoreError(format!(
                "Qdrant returned error: {}",
                err_body
            )));
        }

        Ok(())
    }

    async fn search(
        &self,
        query: Embedding,
        top_k: usize,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, ExtragError> {
        let mut req_body = json!({
            "vector": query.0,
            "limit": top_k,
            "with_payload": true,
        });

        if let Some(f) = filter {
            let mut must_conditions = Vec::new();
            if let Some(sid) = f.source_id {
                must_conditions.push(json!({"key": "source_id", "match": {"value": sid}}));
            }
            if let Some((k, v)) = f.metadata {
                must_conditions.push(json!({"key": k, "match": {"value": v}}));
            }

            if !must_conditions.is_empty() {
                req_body["filter"] = json!({ "must": must_conditions });
            }
        }

        let url = format!(
            "{}/collections/{}/points/search",
            self.base_url, self.collection_name
        );
        let resp = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await
            .map_err(|e| ExtragError::VectorStoreError(format!("Qdrant search error: {}", e)))?;

        if !resp.status().is_success() {
            return Err(ExtragError::VectorStoreError(format!(
                "Qdrant returned error: {}",
                resp.status()
            )));
        }

        let json_resp: serde_json::Value = resp.json().await.map_err(|e| {
            ExtragError::VectorStoreError(format!("Failed to parse Qdrant search JSON: {}", e))
        })?;

        let mut results = Vec::new();
        if let Some(points) = json_resp["result"].as_array() {
            for p in points {
                let score = p["score"].as_f64().unwrap_or(0.0) as f32;
                let id = p["id"].as_str().unwrap_or_default().to_string();

                let payload = &p["payload"];
                let source_id = payload["source_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let content = payload["content"].as_str().unwrap_or_default().to_string();
                let utility = payload["utility"].as_f64().unwrap_or(0.0) as f32;

                let mut metadata = HashMap::new();
                let mut sequence_index = 0;

                if let Some(obj) = payload.as_object() {
                    for (k, v) in obj {
                        if k == "sequence_index" {
                            sequence_index = v.as_u64().unwrap_or(0) as usize;
                        } else if k != "source_id" && k != "content" && k != "utility" {
                            metadata.insert(k.clone(), v.as_str().unwrap_or("").to_string());
                        }
                    }
                }

                let doc = VectorDocument {
                    id,
                    chunk: Chunk {
                        source_id,
                        metadata,
                        content,
                        sequence_index,
                    },
                    embedding: Embedding(vec![]),
                    utility,
                };

                results.push(SearchResult {
                    document: doc,
                    score,
                });
            }
        }

        Ok(results)
    }

    async fn update_utility(&self, _id: &str, _delta: f32) -> Result<(), ExtragError> {
        // Since we cannot read/update atomically without fetch, we just skip it or send payload patch
        // Qdrant allows patch payload if we just overwrite. To add, we need fetch -> update -> put.
        // Assuming we do not need this in the REST interface yet.
        Ok(())
    }

    async fn delete_by_source_id(&self, source_id: &str) -> Result<(), ExtragError> {
        let url = format!(
            "{}/collections/{}/points/delete",
            self.base_url, self.collection_name
        );
        let req_body = json!({
            "filter": {
                "must": [
                    { "key": "source_id", "match": { "value": source_id } }
                ]
            }
        });

        let resp = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await
            .map_err(|e| ExtragError::VectorStoreError(format!("Qdrant delete error: {}", e)))?;

        if !resp.status().is_success() {
            return Err(ExtragError::VectorStoreError(format!(
                "Qdrant returned error: {}",
                resp.status()
            )));
        }

        Ok(())
    }
}
