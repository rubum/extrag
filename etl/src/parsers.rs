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

/// A robust parser for PDF documents using the pdf_oxide library.
///
/// This parser is written in pure Rust and handles complex font encodings
/// like Identity-H that often defeat simpler parsers.
pub struct PdfParser;

impl Parser for PdfParser {
    fn supports(&self, payload: &RawPayload) -> bool {
        payload.format == Format::Pdf
    }

    fn parse(&self, payload: &RawPayload) -> Result<String, ExtragError> {
        use pdf_oxide::document::PdfDocument;

        // 1. Try robust text extraction first (pdf_oxide)
        let mut doc = PdfDocument::from_bytes(payload.content.to_vec())
            .map_err(|e| ExtragError::ParseError(format!("Failed to load PDF document: {}", e)))?;

        let mut full_text = String::new();
        let page_count = doc
            .page_count()
            .map_err(|e| ExtragError::ParseError(format!("Failed to get page count: {}", e)))?;

        for i in 0..page_count {
            if let Ok(text) = doc.extract_text(i) {
                if !text.trim().is_empty() {
                    full_text.push_str(&text);
                    full_text.push_str("\n\n");
                }
            }
        }

        // 2. OCR Fallback if text layer is empty
        if full_text.trim().is_empty() {
            tracing::info!("Text layer is empty. Attempting OCR fallback...");

            use pdfium_render::prelude::*;

            // Re-load with Pdfium for rendering.
            // Note: Pdfium bindings often require 'static lifetimes for the bytes.
            let pdfium = Pdfium::new(Pdfium::bind_to_system_library().map_err(|e| {
                ExtragError::ParseError(format!(
                    "OCR Failed: No pdfium library found for rendering: {}",
                    e
                ))
            })?);

            // Pragmatic fix: Leak a clone of the bytes to satisfy 'static requirement in fallback path.
            let static_bytes: &'static [u8] =
                Box::leak(payload.content.to_vec().into_boxed_slice());

            let doc = pdfium
                .load_pdf_from_byte_slice(static_bytes, None)
                .map_err(|e| {
                    ExtragError::ParseError(format!(
                        "Failed to load PDF in Pdfium for OCR: {:?}",
                        e
                    ))
                })?;

            // In pdfium-render 0.8, pages() returns a collection that provides an iter() method.
            for page in doc.pages().iter() {
                // Render page to image
                let render_config = PdfRenderConfig::new().set_target_width(2000); // Approximate higher resolution

                let bitmap = page.render_with_config(&render_config).map_err(|e| {
                    ExtragError::ParseError(format!("Failed to render page for OCR: {:?}", e))
                })?;

                let _image = bitmap.as_image();

                // OCR logic would go here:
                // if let Ok(ocr_text) = ocr_engine.recognize(&_image) {
                //     full_text.push_str(&ocr_text);
                // }

                tracing::warn!(
                    "OCR Rendering successful for page, but models not yet initialized."
                );
            }
        }

        if full_text.trim().is_empty() {
            return Err(ExtragError::ParseError(
                "PDF parsing resulted in empty text. Document appears to be a scanned image and OCR models are missing.".to_string()
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
