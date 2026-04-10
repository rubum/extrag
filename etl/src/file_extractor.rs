//! # Filesystem Data Extraction
//! 
//! Provides a robust implementation of [`BatchExtractor`] for scanning and reading 
//! files from a local directory.

use async_trait::async_trait;
use bytes::Bytes;
use extrag_core::error::ExtragError;
use extrag_core::etl::BatchExtractor;
use extrag_core::payload::{Format, RawPayload};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

/// A high-performance extractor that recursively reads documents from the local filesystem.
/// 
/// It automatically detects file formats based on extensions and handles metadata 
/// extraction such as modification timestamps for incremental syncs.
pub struct FilesystemExtractor {
    /// The root directory path to scan.
    pub dir_path: PathBuf,
}

impl FilesystemExtractor {
    /// Creates a new FilesystemExtractor pointing to the specified path.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            dir_path: path.as_ref().to_path_buf(),
        }
    }

    /// Determines the [`Format`] of a file based on its filesystem extension.
    fn determine_format(path: &Path) -> Format {
        match path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase()
            .as_str()
        {
            "md" | "markdown" => Format::Markdown,
            "json" => Format::Json,
            "pdf" => Format::Pdf,
            "html" | "htm" => Format::Html,
            _ => Format::PlainText,
        }
    }

    /// Helper to identify hidden files or directories that should be skipped.
    fn is_hidden(entry: &DirEntry) -> bool {
        entry.depth() > 0
            && entry
                .file_name()
                .to_str()
                .map(|s| s.starts_with('.'))
                .unwrap_or(false)
    }
}

#[async_trait]
impl BatchExtractor for FilesystemExtractor {
    /// Scans the directory for new or modified files.
    /// 
    /// # Arguments
    /// * `_since_timestamp` - Reserved for incremental sync optimization.
    /// 
    /// # Errors
    /// Returns [`ExtragError::ConnectionError`] if the directory is missing or inaccessible.
    #[tracing::instrument(skip(self))]
    async fn fetch_batch(
        &self,
        _since_timestamp: Option<i64>,
    ) -> Result<Vec<RawPayload>, ExtragError> {
        if !self.dir_path.exists() || !self.dir_path.is_dir() {
            return Err(ExtragError::ConnectionError(format!(
                "Directory {:?} does not exist or is not a directory",
                self.dir_path
            )));
        }

        tracing::info!("Scanning directory: {:?}", self.dir_path);

        let dir_path = self.dir_path.clone();

        // Run the blocking walkdir logic in a dedicated thread pool
        let file_paths = tokio::task::spawn_blocking(move || {
            let mut paths = Vec::new();
            let walker = WalkDir::new(&dir_path)
                .into_iter()
                .filter_entry(|e| !Self::is_hidden(e));

            for entry in walker.filter_map(|e| e.ok()) {
                if entry.file_type().is_file() {
                    paths.push(entry.path().to_path_buf());
                }
            }
            paths
        })
        .await
        .map_err(|e| ExtragError::ConnectionError(format!("Filesystem scan panic: {}", e)))?;

        tracing::info!("Found {} candidate files for processing", file_paths.len());

        let mut payloads = Vec::new();
        let mut bytes_ingested = 0;
        
        for path in file_paths {
            let content = match tokio::fs::read(&path).await {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("Failed to read file {:?}: {}", path, e);
                    continue;
                }
            };

            bytes_ingested += content.len() as u64;
            let source_id = path.to_string_lossy().to_string();
            let format = Self::determine_format(&path);

            let mut metadata = HashMap::new();
            if let Ok(modified) = tokio::fs::metadata(&path).await.and_then(|m| m.modified()) {
                let duration = modified
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                metadata.insert("last_modified".to_string(), duration.as_secs().to_string());
            }

            payloads.push(RawPayload {
                source_id,
                format,
                content: Bytes::from(content),
                metadata,
            });
        }

        tracing::debug!(
            bytes_ingested = bytes_ingested,
            "Extracted {} payloads totaling {} bytes", payloads.len(), bytes_ingested
        );

        Ok(payloads)
    }
}
