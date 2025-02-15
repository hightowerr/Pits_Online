from llama_index.core import VectorStoreIndex, TreeIndex, load_index_from_storage
from llama_index.core import StorageContext
from global_settings import INDEX_STORAGE
from document_uploader import ingest_documents

def build_indexes(nodes):
    try:
        print(f"Attempting to load indexes from: {INDEX_STORAGE}")
        storage_context = StorageContext.from_defaults(
            persist_dir=INDEX_STORAGE
        )
        
        try:
            vector_index = load_index_from_storage(
                storage_context, index_id="vector"
            )
            print("Vector index loaded successfully from storage.")
        except Exception as vector_load_error:
            print(f"Failed to load vector index: {vector_load_error}")
            vector_index = None

        try:
            tree_index = load_index_from_storage(
                storage_context, index_id="tree"
            )
            print("Tree index loaded successfully from storage.")
        except Exception as tree_load_error:
            print(f"Failed to load tree index: {tree_load_error}")
            tree_index = None

        # If either index failed to load, recreate both
        if vector_index is None or tree_index is None:
            raise Exception("One or more indexes could not be loaded")

    except Exception as e:
        print(f"Comprehensive index loading error: {e}")
        print("Creating new indexes from scratch...")
        
        storage_context = StorageContext.from_defaults()
        
        vector_index = VectorStoreIndex(
            nodes, storage_context=storage_context
        )
        vector_index.set_index_id("vector")
        print("Vector index created.")
        
        tree_index = TreeIndex(
            nodes, storage_context=storage_context
        )
        tree_index.set_index_id("tree")
        print("Tree index created.")
        
        try:
            storage_context.persist(persist_dir=INDEX_STORAGE)
            print(f"Indexes persisted to: {INDEX_STORAGE}")
        except Exception as persist_error:
            print(f"Error persisting indexes: {persist_error}")

    return vector_index, tree_index

# Optional: Add a main block for direct testing
if __name__ == "__main__":
    # Example usage for debugging
    from document_uploader import ingest_documents
    
    print("Starting index builder...")
    nodes = ingest_documents()
    print(f"Ingested {len(nodes)} nodes")
    
    vector_index, tree_index = build_indexes(nodes)
    
    print("Index building complete.")

