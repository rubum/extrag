//! # ETL (Extract, Transform, Load) Foundations
//! 
//! Defines the core traits for fetching data from diverse sources and transforming
//! raw bytes into clean, indexable text.

use crate::error::ExtragError;
use crate::payload::RawPayload;
use async_trait::async_trait;
use futures::stream::BoxStream;

/// Defines a batch-oriented data source (e.g., S3, local filesystem, Postgres polling).
///
/// These extractors are designed to be polled or scheduled to fetch documents
/// since a specific watermark.
#[async_trait]
pub trait BatchExtractor: Send + Sync {
    /// Fetches a batch of payloads modified since a given timestamp.
    ///
    /// # Arguments
    /// * `since_timestamp` - Optional Unix epoch seconds to implement incremental sync.
    ///
    /// # Errors
    /// Returns [`ExtragError::ConnectionError`] if the source is unavailable.
    async fn fetch_batch(
        &self,
        since_timestamp: Option<i64>,
    ) -> Result<Vec<RawPayload>, ExtragError>;
}

/// Defines a continuous, real-time data source (e.g., Kafka, NATS, Webhooks).
///
/// Stream receivers provide immediate updates as they occur in the source system,
/// ensuring the vector store remains synchronized with low latency.
pub trait StreamReceiver: Send + Sync {
    /// Returns an asynchronous stream of raw payloads.
    ///
    /// Implementers should handle internal reconnections and yield `Result` items.
    fn receive_stream(&self) -> BoxStream<'static, Result<RawPayload, ExtragError>>;
}

/// A parser responsible for transforming raw binary data into normalized UTF-8 text.
pub trait Parser: Send + Sync {
    /// Determines if this parser implementation is capable of handling the given payload.
    fn supports(&self, payload: &RawPayload) -> bool;

    /// Converts raw bytes within a [`RawPayload`] into a cleaned text string.
    ///
    /// # Errors
    /// Returns [`ExtragError::ParseError`] if decoding fails or the content is malformed.
    fn parse(&self, payload: &RawPayload) -> Result<String, ExtragError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::Format;
    use bytes::Bytes;
    use std::collections::HashMap;

    /// A mock parser for testing the Parser trait abstraction.
    struct MockMarkdownParser;

    impl Parser for MockMarkdownParser {
        fn supports(&self, payload: &RawPayload) -> bool {
            payload.format == Format::Markdown
        }

        fn parse(&self, payload: &RawPayload) -> Result<String, ExtragError> {
            String::from_utf8(payload.content.to_vec())
                .map_err(|e| ExtragError::ParseError(e.to_string()))
        }
    }

    #[test]
    fn test_markdown_parser_supports_and_parses() {
        let parser = MockMarkdownParser;
        let payload = RawPayload {
            source_id: "example_note.md".into(),
            format: Format::Markdown,
            content: Bytes::from("## Header\nSome markdown details here."),
            metadata: HashMap::new(),
        };

        assert!(
            parser.supports(&payload),
            "Parser should support markdown format"
        );

        let result = parser.parse(&payload).unwrap();
        assert_eq!(result, "## Header\nSome markdown details here.");
    }
}
