//! # Optical Character Recognition (OCR) Module
//!
//! This module provides high-fidelity text extraction from images and scanned documents 
//! using the pure-Rust `ocrs` engine and the `rten` inference runtime.
//!
//! ## Design Architecture
//!
//! The OCR system is designed as a fallback for the ETL pipeline. When a document (like a PDF) 
//! lacks a searchable text layer, the system can rasterize the document into images and 
//! process them through this module.
//!
//! ## Model Requirements
//!
//! The OCR engine requires two pre-trained models in `.rten` format:
//! 1. **Detection Model**: Responsible for identifying bounding boxes of text regions.
//! 2. **Recognition Model**: Responsible for converting the identified regions into characters.
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use etl::ocr::OcrEngineWrapper;
//! use image::open;
//!
//! # fn example() -> anyhow::Result<()> {
//! let det_model = std::fs::read("text-detection.rten")?;
//! let rec_model = std::fs::read("text-recognition.rten")?;
//! 
//! let ocr = OcrEngineWrapper::new(det_model, rec_model)?;
//! let img = open("scanned_page.png")?;
//! 
//! let text = ocr.recognize(&img)?;
//! println!("Extracted text: {}", text);
//! # Ok(())
//! # }
//! ```

use extrag_core::error::ExtragError;
use image::DynamicImage;
use ocrs::{OcrEngine, OcrEngineParams};
use rten::Model;
use rten_tensor::NdTensor;

/// A wrapper around the `ocrs` engine providing a simplified interface 
/// for the Extrag ETL pipeline.
pub struct OcrEngineWrapper {
    /// The underlying pure-Rust OCR engine.
    engine: OcrEngine,
}

impl OcrEngineWrapper {
    /// Creates a new OCR engine.
    ///
    /// Requires pre-trained models in .rten format.
    pub fn new(
        detection_model_data: Vec<u8>,
        recognition_model_data: Vec<u8>,
    ) -> Result<Self, ExtragError> {
        let detection_model = Model::load(&detection_model_data).map_err(|e| {
            ExtragError::ParseError(format!("Failed to load detection model: {}", e))
        })?;
        let recognition_model = Model::load(&recognition_model_data).map_err(|e| {
            ExtragError::ParseError(format!("Failed to load recognition model: {}", e))
        })?;

        let params = OcrEngineParams {
            detection_model: Some(detection_model),
            recognition_model: Some(recognition_model),
            ..Default::default()
        };

        let engine = OcrEngine::new(params).map_err(|e| {
            ExtragError::ParseError(format!("Failed to initialize OCR engine: {}", e))
        })?;

        Ok(Self { engine })
    }

    /// Recognizes text in a dynamic image.
    pub fn recognize(&self, image: &DynamicImage) -> Result<String, ExtragError> {
        let img_rgb = image.to_rgb8();
        let (width, height) = img_rgb.dimensions();

        // Convert to CHW f32 tensor [3, H, W] normalized to [-0.5, 0.5] as expected by many models
        // ocrs prepare_input handles some normalization but it's best to provide a standard tensor.
        let mut tensor = NdTensor::zeros([3, height as usize, width as usize]);
        for y in 0..height {
            for x in 0..width {
                let pixel = img_rgb.get_pixel(x, y);
                let [r, g, b] = pixel.0;
                tensor[[0, y as usize, x as usize]] = (r as f32) / 255.0;
                tensor[[1, y as usize, x as usize]] = (g as f32) / 255.0;
                tensor[[2, y as usize, x as usize]] = (b as f32) / 255.0;
            }
        }

        const ERROR_MSG: &str = "OCR process failed";

        let input = self
            .engine
            .prepare_input(tensor.view())
            .map_err(|e| ExtragError::ParseError(format!("{}: {:?}", ERROR_MSG, e)))?;

        let text = self
            .engine
            .get_text(&input)
            .map_err(|e| ExtragError::ParseError(format!("{}: {:?}", ERROR_MSG, e)))?;

        Ok(text)
    }
}
