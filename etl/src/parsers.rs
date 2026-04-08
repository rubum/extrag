//! # Content Parsers
//! 
//! Provides specialized implementations of [`Parser`] to transform various raw 
//! file formats into indexable UTF-8 strings.

use extrag_core::error::ExtragError;
use extrag_core::etl::Parser;
use extrag_core::payload::{Format, RawPayload};

/// A robust parser for Markdown documents.
/// 
/// Currently performs basic UTF-8 validation and extraction. Future iterations 
/// may include frontmatter stripping or structural awareness.
pub struct MarkdownParser;

impl Parser for MarkdownParser {
    fn supports(&self, payload: &RawPayload) -> bool {
        payload.format == Format::Markdown
    }

    fn parse(&self, payload: &RawPayload) -> Result<String, ExtragError> {
        String::from_utf8(payload.content.to_vec())
            .map_err(|e| ExtragError::ParseError(format!("Invalid UTF-8 in Markdown: {}", e)))
    }
}

/// A parser for JSON structured data.
/// 
/// Validates the JSON integrity and returns a pretty-printed string representation 
/// to ensure the chunker can maintain structural hierarchy where possible.
pub struct JsonParser;

impl Parser for JsonParser {
    fn supports(&self, payload: &RawPayload) -> bool {
        payload.format == Format::Json
    }

    fn parse(&self, payload: &RawPayload) -> Result<String, ExtragError> {
        let text = String::from_utf8(payload.content.to_vec())
            .map_err(|e| ExtragError::ParseError(format!("Invalid UTF-8 in JSON: {}", e)))?;

        // Ensure it is valid JSON
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ExtragError::ParseError(format!("Invalid JSON format: {}", e)))?;

        // Stringify neatly for chunking
        Ok(serde_json::to_string_pretty(&value).unwrap_or(text))
    }
}

/// A standard parser for Plain Text as a fallback.
pub struct PlainTextParser;

impl Parser for PlainTextParser {
    fn supports(&self, payload: &RawPayload) -> bool {
        payload.format == Format::PlainText
    }

    fn parse(&self, payload: &RawPayload) -> Result<String, ExtragError> {
        String::from_utf8(payload.content.to_vec())
            .map_err(|e| ExtragError::ParseError(format!("Invalid UTF-8 in PlainText: {}", e)))
    }
}
