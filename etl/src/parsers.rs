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

/// A robust parser for PDF documents using the lopdf library.
pub struct PdfParser;

impl Parser for PdfParser {
    fn supports(&self, payload: &RawPayload) -> bool {
        payload.format == Format::Pdf
    }

    fn parse(&self, payload: &RawPayload) -> Result<String, ExtragError> {
        use lopdf::Document;

        let doc = Document::load_mem(&payload.content)
            .map_err(|e| ExtragError::ParseError(format!("Failed to load PDF document: {}", e)))?;

        let mut full_text = String::new();

        // Iterate through all pages and extract text content
        let page_numbers: Vec<u32> = doc.get_pages().keys().cloned().collect();
        for page_num in page_numbers {
            match doc.extract_text(&[page_num]) {
                Ok(text) => {
                    full_text.push_str(&text);
                    full_text.push_str("\n\n"); // Maintain page separation
                }
                Err(e) => {
                    tracing::warn!("Failed to extract text from PDF page {}: {}", page_num, e);
                    continue;
                }
            }
        }

        if full_text.trim().is_empty() {
            return Err(ExtragError::ParseError(
                "PDF parsing resulted in empty text. It might be an image-only (scanned) PDF."
                    .to_string(),
            ));
        }

        Ok(full_text)
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

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use extrag_core::payload::Format;
    use lopdf::{Dictionary, Document, Object, Stream};
    use std::collections::HashMap;

    #[test]
    fn test_pdf_parser_extraction() {
        // Create a minimal programmatic PDF for testing
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(Dictionary::from_iter(vec![
            ("Type", Object::Name("Font".into())),
            ("Subtype", Object::Name("Type1".into())),
            ("BaseFont", Object::Name("Helvetica".into())),
        ]));
        let resources_id = doc.add_object(Dictionary::from_iter(vec![(
            "Font",
            Object::Reference(font_id).into(),
        )]));
        let content = "BT /F1 12 Tf 100 700 Td (Hello Extrag PDF) Tj ET";
        let content_id =
            doc.add_object(Stream::new(Dictionary::new(), content.as_bytes().to_vec()));
        let page_id = doc.add_object(Dictionary::from_iter(vec![
            ("Type", Object::Name("Page".into())),
            ("Parent", Object::Reference(pages_id)),
            ("Contents", Object::Reference(content_id)),
            ("Resources", Object::Reference(resources_id)),
            (
                "MediaBox",
                vec![0.into(), 0.into(), 600.into(), 800.into()].into(),
            ),
        ]));
        let pages = Dictionary::from_iter(vec![
            ("Type", Object::Name("Pages".into())),
            ("Kids", vec![Object::Reference(page_id)].into()),
            ("Count", 1.into()),
        ]);
        doc.objects.insert(pages_id, Object::Dictionary(pages));
        let catalog_id = doc.add_object(Dictionary::from_iter(vec![
            ("Type", Object::Name("Catalog".into())),
            ("Pages", Object::Reference(pages_id)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog_id));

        let mut buffer = Vec::new();
        doc.save_to(&mut buffer).unwrap();

        let parser = PdfParser;
        let payload = RawPayload {
            source_id: "test.pdf".into(),
            format: Format::Pdf,
            content: Bytes::from(buffer),
            metadata: HashMap::new(),
        };

        let result = parser.parse(&payload).expect("Failed to parse PDF");
        assert!(result.contains("Hello Extrag PDF"));
    }

    #[test]
    fn test_pdf_parser_empty_error() {
        let parser = PdfParser;
        // Create a valid PDF but with no text content
        let mut doc = Document::with_version("1.5");
        let mut buffer = Vec::new();
        doc.save_to(&mut buffer).unwrap();

        let payload = RawPayload {
            source_id: "empty.pdf".into(),
            format: Format::Pdf,
            content: Bytes::from(buffer),
            metadata: HashMap::new(),
        };

        let result = parser.parse(&payload);
        assert!(result.is_err());
    }
}
