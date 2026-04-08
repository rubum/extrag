//! # Text Chunking
//! 
//! Provides strategies for breaking down large documents into manageable pieces
//! for vector embedding and retrieval. Supports character-based, token-based,
//! and recursive splitting.

use crate::error::ExtragError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tiktoken_rs::CoreBPE;

/// Default separators used by the [`RecursiveCharacterChunker`].
pub const DEFAULT_SEPARATORS: &[&str] = &["\n\n", "\n", " ", ""];

/// A unit of text extracted and chunked from a larger document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// The unique identifier of the source document.
    pub source_id: String,

    /// The actual text content of this chunk.
    pub content: String,

    /// Metadata associated with this specific chunk (e.g., page number, section).
    pub metadata: HashMap<String, String>,

    /// The sequential position of this chunk within the source document.
    pub sequence_index: usize,
}

/// A trait for splitting a large document (parsed text) into smaller [`Chunk`] units.
pub trait Chunker: Send + Sync {
    /// Splits the input text into a vector of Chunks.
    ///
    /// # Example
    /// ```
    /// use extrag_core::chunker::{Chunker, CharacterChunker};
    /// let chunker = CharacterChunker { chunk_size: 10, overlap: 0 };
    /// let chunks = chunker.chunk("doc_1", "Hello World!").unwrap();
    /// assert_eq!(chunks.len(), 2);
    /// ```
    ///
    /// # Errors
    /// Returns [`ExtragError::ChunkingError`] if parameters (like chunk size) are invalid.
    fn chunk(&self, source_id: &str, text: &str) -> Result<Vec<Chunk>, ExtragError>;
}

/// A simple chunker that splits text by a fixed character count.
///
/// > [!WARNING]
/// > Character-based splitting is less precise for LLM context windows than token-based splitting.
pub struct CharacterChunker {
    /// The maximum number of characters per chunk.
    pub chunk_size: usize,
    /// The number of characters to overlap between adjacent chunks.
    pub overlap: usize,
}

impl Chunker for CharacterChunker {
    fn chunk(&self, source_id: &str, text: &str) -> Result<Vec<Chunk>, ExtragError> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;
        let mut index = 0;

        if self.chunk_size == 0 {
            return Err(ExtragError::ChunkingError(
                "Chunk size must be greater than zero".into(),
            ));
        }

        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());
            let content: String = chars[start..end].iter().collect();

            chunks.push(Chunk {
                source_id: source_id.to_string(),
                content,
                metadata: HashMap::new(),
                sequence_index: index,
            });

            index += 1;
            if end == chars.len() {
                break;
            }

            // Advance by (chunk_size - overlap)
            let step = if self.overlap < self.chunk_size {
                self.chunk_size - self.overlap
            } else {
                1 // Ensure forward progress
            };
            start += step;
        }

        Ok(chunks)
    }
}

/// A chunker that splits text based on the number of tokens,
/// rather than characters, ensuring compatibility with LLM context windows.
pub struct TokenChunker {
    /// The BPE tokenizer instance.
    pub bpe: CoreBPE,
    /// The maximum number of tokens per chunk.
    pub chunk_size: usize,
    /// The number of tokens to overlap between adjacent chunks.
    pub overlap: usize,
}

impl TokenChunker {
    /// Creates a new TokenChunker using the standard `cl100k_base` encoding (used by GPT-4).
    ///
    /// # Errors
    /// Returns an error if the tokenizer fails to load.
    pub fn new(chunk_size: usize, overlap: usize) -> Result<Self, ExtragError> {
        let bpe =
            tiktoken_rs::cl100k_base().map_err(|e| ExtragError::ChunkingError(e.to_string()))?;
        Ok(Self {
            bpe,
            chunk_size,
            overlap,
        })
    }
}

impl Chunker for TokenChunker {
    fn chunk(&self, source_id: &str, text: &str) -> Result<Vec<Chunk>, ExtragError> {
        if self.chunk_size == 0 {
            return Err(ExtragError::ChunkingError(
                "Chunk size must be greater than zero".into(),
            ));
        }

        let tokens = self.bpe.encode_ordinary(text);
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut index = 0;

        while start < tokens.len() {
            let end = (start + self.chunk_size).min(tokens.len());
            let chunk_tokens = &tokens[start..end];

            // tiktoken-rs decode returns a Result<String, _>
            let content = self
                .bpe
                .decode(chunk_tokens.to_vec())
                .map_err(|e| ExtragError::ChunkingError(e.to_string()))?;

            chunks.push(Chunk {
                source_id: source_id.to_string(),
                content,
                metadata: HashMap::new(),
                sequence_index: index,
            });

            index += 1;
            if end == tokens.len() {
                break;
            }

            let step = if self.overlap < self.chunk_size {
                self.chunk_size - self.overlap
            } else {
                1
            };
            start += step;
        }

        Ok(chunks)
    }
}

/// A robust chunker that attempts to split text at natural boundaries (paragraphs, sentences)
/// before falling back to character splitting.
///
/// This is the recommended chunker for most RAG applications as it preserves semantic integrity.
pub struct RecursiveCharacterChunker {
    /// The target maximum size for each chunk (in characters).
    pub chunk_size: usize,
    /// The overlap between chunks (in characters).
    pub overlap: usize,
    /// Ordered list of separators to try splitting on.
    pub separators: Vec<String>,
}

impl Default for RecursiveCharacterChunker {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            overlap: 200,
            separators: DEFAULT_SEPARATORS.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl RecursiveCharacterChunker {
    /// Recursively splits the text and merges chunks up to the chunk_size limit.
    fn split_and_merge(&self, text: &str, sep_idx: usize) -> Vec<String> {
        let text_len = text.chars().count();
        if text_len <= self.chunk_size {
            return vec![text.to_string()];
        }

        let sep = if sep_idx < self.separators.len() {
            &self.separators[sep_idx]
        } else {
            ""
        };

        let splits: Vec<&str> = if sep.is_empty() {
            // filter out empty splits from `.split("")`
            text.split("").filter(|s| !s.is_empty()).collect()
        } else {
            text.split(sep).collect()
        };

        let mut merged = Vec::new();
        let mut current_chunk = String::new();

        for split in splits {
            let sep_len = if current_chunk.is_empty() || sep.is_empty() {
                0
            } else {
                sep.chars().count()
            };
            let part_len = split.chars().count();
            let proposed_len = current_chunk.chars().count() + sep_len + part_len;

            if proposed_len > self.chunk_size && !current_chunk.is_empty() {
                merged.push(current_chunk.clone());
                // naive overlap - start fresh
                current_chunk = split.to_string();
            } else {
                if !current_chunk.is_empty() && !sep.is_empty() {
                    current_chunk.push_str(sep);
                }
                current_chunk.push_str(split);
            }
        }

        if !current_chunk.is_empty() {
            merged.push(current_chunk);
        }

        let mut final_chunks = Vec::new();
        for chunk in merged {
            if chunk.chars().count() > self.chunk_size && sep_idx + 1 < self.separators.len() {
                final_chunks.extend(self.split_and_merge(&chunk, sep_idx + 1));
            } else if chunk.chars().count() > self.chunk_size {
                // If it's still too large but we're at the last separator, force character split
                final_chunks.extend(self.split_and_merge(&chunk, self.separators.len() - 1));
            } else {
                final_chunks.push(chunk);
            }
        }

        final_chunks
    }
}

impl Chunker for RecursiveCharacterChunker {
    fn chunk(&self, source_id: &str, text: &str) -> Result<Vec<Chunk>, ExtragError> {
        if self.chunk_size == 0 {
            return Err(ExtragError::ChunkingError("Chunk size must be > 0".into()));
        }

        let texts = self.split_and_merge(text, 0);
        let mut chunks = Vec::new();
        for (i, t) in texts.into_iter().enumerate() {
            chunks.push(Chunk {
                source_id: source_id.to_string(),
                content: t,
                metadata: HashMap::new(),
                sequence_index: i,
            });
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_chunker() {
        let chunker = CharacterChunker {
            chunk_size: 10,
            overlap: 2,
        };
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = chunker.chunk("test_doc", text).unwrap();

        assert_eq!(chunks[0].content, "abcdefghij");
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_token_chunker() {
        let chunker = TokenChunker::new(5, 1).unwrap();
        let text = "Hello world! This is a test of the token chunker.";
        let chunks = chunker.chunk("token_doc", text).unwrap();

        assert!(!chunks.is_empty());
        // Check that decoding the chunk gives a valid string
        assert!(chunks[0].content.contains("Hello"));
    }

    #[test]
    fn test_recursive_chunker() {
        let chunker = RecursiveCharacterChunker {
            chunk_size: 50,
            overlap: 0,
            ..Default::default()
        };

        let text = "Paragraph one.\n\nParagraph two is slightly longer.\n\nParagraph three.";
        let chunks = chunker.chunk("rec_doc", text).unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0].content,
            "Paragraph one.\n\nParagraph two is slightly longer."
        );
        assert_eq!(chunks[1].content, "Paragraph three.");
    }
}
