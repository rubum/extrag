//! # Sync State Management
//!
//! Provides persistent storage for document synchronization states.
//! This allows the pipeline to perform delta extractions by tracking
//! file hashes and modification timestamps.

use async_trait::async_trait;
use extrag_core::error::ExtragError;
use sqlx::{
    Row,
    sqlite::{SqliteConnectOptions, SqlitePool},
};
use std::path::Path;

/// The SQL schema used to initialize the SQLite state database.
const INIT_SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS document_states (
        source_id TEXT PRIMARY KEY,
        last_modified INTEGER,
        content_hash TEXT
    );
";

/// Represents the persisted state of a document in the ingestion pipeline.
#[derive(Debug, Clone, Default)]
pub struct DocumentSyncState {
    /// The unique identifier of the source document.
    pub source_id: String,
    /// Last seen modification timestamp (Unix epoch).
    pub last_modified: Option<i64>,
    /// cryptographic hash of the document content to detect changes.
    pub content_hash: Option<String>,
}

/// Trait for persisting and retrieving document synchronization metadata.
#[async_trait]
pub trait SyncStateStore: Send + Sync {
    /// Retrieves the recorded state for a document by its source ID.
    async fn get_document_state(
        &self,
        source_id: &str,
    ) -> Result<Option<DocumentSyncState>, ExtragError>;

    /// Records or updates the state for a document.
    async fn update_document_state(&self, state: DocumentSyncState) -> Result<(), ExtragError>;

    /// Removes a document state from the store.
    async fn remove_document_state(&self, source_id: &str) -> Result<(), ExtragError>;

    /// Returns a list of all source IDs currently tracked in the state store.
    async fn get_all_source_ids(&self) -> Result<Vec<String>, ExtragError>;

    /// Clears all stored document states, resetting the synchronization history.
    async fn clear_all(&self) -> Result<(), ExtragError>;
}

/// A SQLite-backed implementation of [`SyncStateStore`].
///
/// Uses a local file to maintain synchronization history across platform restarts.
#[derive(Clone)]
pub struct SqliteSyncStateStore {
    pool: SqlitePool,
}

impl SqliteSyncStateStore {
    /// Creates a new state store at the specified filesystem path.
    ///
    /// # Arguments
    /// * `path` - Path to the `.db` file. Will be created if it does not exist.
    ///
    /// # Errors
    /// Returns [`ExtragError::ConnectionError`] if the database cannot be initialized.
    pub async fn new<P: AsRef<Path>>(path: P) -> Result<Self, ExtragError> {
        let options = SqliteConnectOptions::new()
            .filename(path.as_ref())
            .create_if_missing(true);

        let pool = SqlitePool::connect_with(options).await.map_err(|e| {
            ExtragError::ConnectionError(format!("Failed to connect to SQLite: {}", e))
        })?;

        // Initialize schema
        sqlx::query(INIT_SCHEMA).execute(&pool).await.map_err(|e| {
            ExtragError::ConnectionError(format!("Failed to initialize sync schema: {}", e))
        })?;

        Ok(Self { pool })
    }
}

#[async_trait]
impl SyncStateStore for SqliteSyncStateStore {
    async fn get_document_state(
        &self,
        source_id: &str,
    ) -> Result<Option<DocumentSyncState>, ExtragError> {
        let row = sqlx::query("SELECT source_id, last_modified, content_hash FROM document_states WHERE source_id = ?")
            .bind(source_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| ExtragError::VectorStoreError(format!("DB query failed: {}", e)))?;

        Ok(row.map(|r| DocumentSyncState {
            source_id: r.get(0),
            last_modified: r.get(1),
            content_hash: r.get(2),
        }))
    }

    async fn update_document_state(&self, state: DocumentSyncState) -> Result<(), ExtragError> {
        sqlx::query("INSERT OR REPLACE INTO document_states (source_id, last_modified, content_hash) VALUES (?, ?, ?)")
            .bind(state.source_id)
            .bind(state.last_modified)
            .bind(state.content_hash)
            .execute(&self.pool)
            .await
            .map_err(|e| ExtragError::VectorStoreError(format!("DB insert failed: {}", e)))?;
        Ok(())
    }

    async fn remove_document_state(&self, source_id: &str) -> Result<(), ExtragError> {
        sqlx::query("DELETE FROM document_states WHERE source_id = ?")
            .bind(source_id)
            .execute(&self.pool)
            .await
            .map_err(|e| ExtragError::VectorStoreError(format!("DB delete failed: {}", e)))?;
        Ok(())
    }

    async fn get_all_source_ids(&self) -> Result<Vec<String>, ExtragError> {
        let rows = sqlx::query("SELECT source_id FROM document_states")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| ExtragError::VectorStoreError(format!("DB selection failed: {}", e)))?;

        Ok(rows.into_iter().map(|r| r.get(0)).collect())
    }

    async fn clear_all(&self) -> Result<(), ExtragError> {
        sqlx::query("DELETE FROM document_states")
            .execute(&self.pool)
            .await
            .map_err(|e| ExtragError::VectorStoreError(format!("DB clear failed: {}", e)))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_sqlite_sync_state_clear_all() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_sync.db");
        let store = SqliteSyncStateStore::new(&db_path).await.unwrap();

        // 1. Initial state should be empty
        assert_eq!(store.get_all_source_ids().await.unwrap().len(), 0);

        // 2. Add some states
        store
            .update_document_state(DocumentSyncState {
                source_id: "doc1".to_string(),
                last_modified: Some(100),
                content_hash: Some("hash1".to_string()),
            })
            .await
            .unwrap();

        store
            .update_document_state(DocumentSyncState {
                source_id: "doc2".to_string(),
                last_modified: Some(200),
                content_hash: Some("hash2".to_string()),
            })
            .await
            .unwrap();

        assert_eq!(store.get_all_source_ids().await.unwrap().len(), 2);

        // 3. Clear all
        store.clear_all().await.unwrap();

        // 4. Should be empty again
        assert_eq!(store.get_all_source_ids().await.unwrap().len(), 0);
    }
}
