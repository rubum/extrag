#[cfg(test)]
mod tests {
    use crate::chunker::Chunk;
    use crate::embeddings::Embedding;
    use crate::test_utils::InMemoryVectorStore;
    use crate::vector_store::{VectorDocument, VectorStore};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_memrl_ema_update() {
        let store = InMemoryVectorStore::new();

        let doc_id = "test_doc".to_string();
        let doc = VectorDocument {
            id: doc_id.clone(),
            chunk: Chunk {
                source_id: "src".into(),
                metadata: HashMap::new(),
                content: "content".into(),
                sequence_index: 0,
            },
            embedding: Embedding(vec![0.1; 384]),
            utility: 0.0,
        };

        store.index(vec![doc]).await.unwrap();

        // Apply feedback (+1.0)
        // new_val = 0.0 + 0.1 * (1.0 - 0.0) = 0.1
        store.update_utility(&doc_id, 1.0).await.unwrap();

        let results = store
            .search(Embedding(vec![0.1; 384]), 1, None)
            .await
            .unwrap();
        assert!((results[0].document.utility - 0.1).abs() < 0.001);

        // Apply another feedback (+1.0)
        // new_val = 0.1 + 0.1 * (1.0 - 0.1) = 0.1 + 0.09 = 0.19
        store.update_utility(&doc_id, 1.0).await.unwrap();

        let results = store
            .search(Embedding(vec![0.1; 384]), 1, None)
            .await
            .unwrap();
        assert!((results[0].document.utility - 0.19).abs() < 0.001);

        // Apply negative feedback (-1.0)
        // new_val = 0.19 + 0.1 * (-1.0 - 0.19) = 0.19 + 0.1 * (-1.19) = 0.19 - 0.119 = 0.071
        store.update_utility(&doc_id, -1.0).await.unwrap();

        let results = store
            .search(Embedding(vec![0.1; 384]), 1, None)
            .await
            .unwrap();
        assert!((results[0].document.utility - 0.071).abs() < 0.001);
    }
}
