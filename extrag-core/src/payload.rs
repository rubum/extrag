//! # Payload Protocols
//! 
//! Defines the standard interface for raw data as it enters the Extrag system.
//! Payloads encapsulate binary content, metadata, and format descriptors.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported file and data formats handled by the Extrag ETL layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Format {
    Pdf,
    Markdown,
    Html,
    Json,
    PlainText,
}

/// A discrete unit of data ingested from an external source.
/// 
/// RawPayload is the primary input to the Extrag pipeline, representing an 
/// unprocessed document or event prior to parsing and chunking.
#[derive(Debug, Clone)]
pub struct RawPayload {
    /// A unique identifier for the source (e.g., a file path, URL, or database primary key).
    pub source_id: String,

    /// The structured format of the [`Self::content`], used for parser routing.
    pub format: Format,

    /// The raw binary content of the payload.
    pub content: Bytes,

    /// Arbitrary metadata associated with the document (e.g., timestamps, authorship).
    pub metadata: HashMap<String, String>,
}
