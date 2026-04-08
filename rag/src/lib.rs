//! # Extrag Agentic RAG Pipeline
//! 
//! Provides the top-level orchestrators for Extrag. 
//! Contains the `IngestionPipeline` for handling delta extractions and updates, 
//! as well as the `AdvancedRetrievalEngine` for conducting MemRL value-based contextual retrieval.

pub mod ingestion;
pub mod retrieval;
pub mod sync_state;

// Expose main engine logic
pub use ingestion::IngestionPipeline;
pub use retrieval::{AdvancedRetrievalEngine, RetrievalConfig};
pub use sync_state::{DocumentSyncState, SqliteSyncStateStore, SyncStateStore};

#[cfg(test)]
mod tests {
    use super::ingestion::IngestionPipeline;
    use crate::sync_state::SqliteSyncStateStore;
    use etl::file_extractor::FilesystemExtractor;
    use etl::parsers::{JsonParser, MarkdownParser};
    use extrag_core::chunker::RecursiveCharacterChunker;
    use extrag_core::embeddings::Embedding;
    use extrag_core::etl::{BatchExtractor, Parser};
    use extrag_core::test_utils::{InMemoryVectorStore, MockEmbedder};
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_full_ingestion_pipeline() {
        // 1. Setup a temporary directory with mock files
        let dir = tempdir().unwrap();
        let md_path = dir.path().join("test.md");
        let json_path = dir.path().join("data.json");

        let mut md_file = File::create(&md_path).unwrap();
        writeln!(
            md_file,
            "# Test Document\n\nThis is a mock markdown file for integration testing."
        )
        .unwrap();

        let mut json_file = File::create(&json_path).unwrap();
        writeln!(
            json_file,
            "{{\"key\": \"value\", \"description\": \"A JSON mock object\"}}"
        )
        .unwrap();

        // 2. Initialize RAG Components
        let extractor = FilesystemExtractor::new(dir.path());
        let md_parser = MarkdownParser;
        let json_parser = JsonParser;

        let chunker = RecursiveCharacterChunker {
            chunk_size: 50,
            overlap: 10,
            ..Default::default()
        };

        let embedder = MockEmbedder { dimension: 4 };
        let vector_store = InMemoryVectorStore::new();

        // 3. Assemble Pipeline
        let extractors: Vec<Box<dyn BatchExtractor>> = vec![Box::new(extractor)];
        let parsers: Vec<Box<dyn Parser>> = vec![Box::new(md_parser), Box::new(json_parser)];

        let pipeline = IngestionPipeline {
            extractors,
            parsers,
            chunker: Box::new(chunker),
            embedder: Box::new(embedder),
            vector_store: Box::new(vector_store),
            sync_state: None,
            chunk_batch_size: 100,
        };

        // 4. Run pipeline
        let count = pipeline.run().await.expect("Pipeline ran successfully");

        assert!(count > 0);

        // Let's verify Search works
        let query_embedding = Embedding(vec![10.0, 0.0, 0.0, 0.0]); // Matches length based on MockEmbedder
        let results = pipeline
            .vector_store
            .search(query_embedding, 2, None)
            .await
            .unwrap();

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_delta_ingestion_with_sqlite() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("sync.db");
        let md_path = dir.path().join("sync_test.md");

        fs::write(&md_path, "Initial content for sync test.").unwrap();

        let store = SqliteSyncStateStore::new(&db_path).await.unwrap();
        let vector_store = InMemoryVectorStore::new();

        let pipeline = IngestionPipeline {
            extractors: vec![Box::new(FilesystemExtractor::new(dir.path()))],
            parsers: vec![Box::new(MarkdownParser)],
            chunker: Box::new(RecursiveCharacterChunker {
                chunk_size: 100,
                ..Default::default()
            }),
            embedder: Box::new(MockEmbedder { dimension: 4 }),
            vector_store: Box::new(vector_store),
            sync_state: Some(Box::new(store)),
            chunk_batch_size: 10,
        };

        // First run: Should ingest
        let count1 = pipeline.run().await.unwrap();
        assert_eq!(count1, 1);

        // Second run: No changes, should skip
        let count2 = pipeline.run().await.unwrap();
        assert_eq!(count2, 0);

        // Modify file: Should re-ingest
        fs::write(&md_path, "Updated content for sync test. It is now longer.").unwrap();
        let count3 = pipeline.run().await.unwrap();
        assert_eq!(count3, 1);
    }
}
