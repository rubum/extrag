//! # Ollama Backend Integration
//!
//! High-performance client for interacting with the Ollama REST API.
//! Supports both text generation and vector embedding generation.

use crate::embeddings::{Embedder, Embedding};
use crate::error::ExtragError;
use crate::llm::LlmClient;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// The default model used if none is provided during initialization.
pub const DEFAULT_OLLAMA_MODEL: &str = "gemma4:latest";

/// A specialized client for interacting with a localized Ollama instance.
///
/// This client implements both [`Embedder`] and [`LlmClient`], acting as a
/// unified interface for cross-backend AI workflows.
#[derive(Debug, Clone)]
pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
}

impl OllamaClient {
    /// Creates a new OllamaClient pointing to the specified base URL.
    ///
    /// # Arguments
    /// * `base_url` - The endpoint of the Ollama server (e.g., "http://localhost:11434").
    /// * `model` - Optional model name override. Defaults to [`DEFAULT_OLLAMA_MODEL`].
    pub fn new(base_url: impl Into<String>, model: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            model: model.unwrap_or_else(|| DEFAULT_OLLAMA_MODEL.to_string()),
        }
    }
}

#[derive(Serialize)]
struct OllamaGenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    system: Option<&'a str>,
    stream: bool,
    think: bool,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

#[async_trait]
impl LlmClient for OllamaClient {
    async fn generate(&self, prompt: &str) -> Result<String, ExtragError> {
        self.generate_with_system("", prompt).await
    }

    async fn generate_with_system(
        &self,
        system: &str,
        prompt: &str,
    ) -> Result<String, ExtragError> {
        let req = OllamaGenerateRequest {
            model: &self.model,
            prompt,
            system: if system.is_empty() {
                None
            } else {
                Some(system)
            },
            stream: false,
            think: false,
        };

        let url = format!("{}/api/generate", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| ExtragError::LlmError(format!("Ollama generate request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(ExtragError::LlmError(format!(
                "Ollama returned error: {}",
                resp.status()
            )));
        }

        let result: OllamaGenerateResponse = resp.json().await.map_err(|e| {
            ExtragError::LlmError(format!("Failed to parse Ollama response: {}", e))
        })?;

        Ok(result.response)
    }
}

#[async_trait]
impl Embedder for OllamaClient {
    async fn embed(&self, text: &str) -> Result<Embedding, ExtragError> {
        let req = OllamaEmbedRequest {
            model: &self.model,
            prompt: text,
        };

        let url = format!("{}/api/embeddings", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| {
                ExtragError::EmbeddingError(format!("Ollama embed request failed: {}", e))
            })?;

        if !resp.status().is_success() {
            return Err(ExtragError::EmbeddingError(format!(
                "Ollama returned error: {}",
                resp.status()
            )));
        }

        let result: OllamaEmbedResponse = resp.json().await.map_err(|e| {
            ExtragError::EmbeddingError(format!("Failed to parse Ollama response: {}", e))
        })?;

        Ok(Embedding(result.embedding))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>, ExtragError> {
        // Ollama /api/embeddings only supports one at a time currently
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        // For Nomic etc. this varies. We will return 0 to indicate unknown/dynamic.
        // If we strictly wanted to enforce it, we could query it.
        0
    }
}
