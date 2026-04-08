//! # Extrag ETL Library
//! 
//! Provides specialized data extractors and parsers to feed the Extrag 
//! ingestion pipeline.

pub mod file_extractor;
pub mod parsers;

pub use file_extractor::FilesystemExtractor;
pub use parsers::{JsonParser, MarkdownParser, PlainTextParser};
