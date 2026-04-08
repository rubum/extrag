//! # LLM (Large Language Model) Inference
//! 
//! Provides abstractions for interacting with generative AI models. 
//! These traits facilitate advanced retrieval techniques like HyDE and Multi-Query expansion.

use crate::error::ExtragError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// System message or prompt configuration for generating text.
/// 
/// Encapsulates both the persona instructions (system) and the user's specific request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Global instructions for the model (e.g., "You are a helpful assistant").
    pub system_prompt: String,
    /// The actual query or task provided by the user.
    pub user_prompt: String,
}

impl PromptTemplate {
    /// Creates a new prompt template from system and user strings.
    pub fn new(system_prompt: impl Into<String>, user_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            user_prompt: user_prompt.into(),
        }
    }
}

/// Abstract trait for communicating with Large Language Models.
/// 
/// Provides high-level methods for generating context-aware text, generating 
/// hypothetical documents (HyDE), and expanding queries.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Generates a response from the LLM based on a raw prompt string.
    async fn generate(&self, prompt: &str) -> Result<String, ExtragError>;

    /// Generates a response with specific system instructions and a user prompt.
    async fn generate_with_system(&self, system: &str, prompt: &str)
        -> Result<String, ExtragError>;

    /// Specifically used for Hypothetical Document Embeddings (HyDE).
    /// 
    /// Generates a grounded, factual hypothetical paragraph that answers the query
    /// to improve dense vector retrieval performance.
    async fn generate_hypothetical_document(&self, query: &str) -> Result<String, ExtragError> {
        let system_prompt = "You are an expert answering questions. Generate a hypothetical, factual paragraph that directly answers the user's question. Focus on creating context rich in relevant keywords.";
        self.generate_with_system(system_prompt, query).await
    }

    /// Specifically used for Multi-Query Expansion.
    /// 
    /// Generates multiple semantically similar variations of the original query 
    /// to increase search recall across the vector store.
    async fn generate_query_variations(
        &self,
        query: &str,
        num_variations: usize,
    ) -> Result<Vec<String>, ExtragError> {
        let system_prompt = format!("You are an AI assistant tasked with expanding search queries. Generate {} distinct but semantically equivalent variations of the user's query. Output them as a raw JSON array of strings without formatting.", num_variations);

        let json_text = self.generate_with_system(&system_prompt, query).await?;
        let clean_json = json_text
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let variations: Vec<String> = serde_json::from_str(clean_json).map_err(|e| {
            ExtragError::ParseError(format!("Failed to parse query variations from LLM: {}", e))
        })?;

        Ok(variations)
    }
}
