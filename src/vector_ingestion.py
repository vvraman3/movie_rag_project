import json
import os
import chromadb
from chromadb.utils import embedding_functions

# Configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
metadata_file = os.path.join(base_dir, "..", "data", "video_chunk_metadata.json")
db_path = os.path.join(base_dir, "..", "vector_db")

def ingest_metadata():
    if not os.path.exists(metadata_file):
        print(f"Error: {metadata_file} not found.")
        return

    chroma_client = chromadb.PersistentClient(path=db_path)
    # Using the exact same model name as the app
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    
    collection = chroma_client.get_or_create_collection(
        name="movie_scenes", 
        embedding_function=emb_func,
        metadata={"hnsw:space": "cosine"}
    )

    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents, metadatas, ids = [], [], []

    for i, entry in enumerate(data):
        # Create the searchable blob
        rich_context = f"Movie: {entry['movie_name']}. Content: {entry['video_content']}. Summary: {entry['summary_en']}. Emotions: {', '.join(entry['emotions'])}. Sounds: {', '.join(entry['acoustic_tags'])}"

        # SYNC FIX: Ensure keys match exactly what streamlit_app.py calls
        metadata_fields = {
            "movie_name": str(entry['movie_name']).lower(), # Lowercase for easier filtering
            "video_path": str(entry['video_file_path']),
            "start_time": float(entry['start_timestamp']),
            "end_time": float(entry['end_timestamp'])
        }

        documents.append(rich_context)
        metadatas.append(metadata_fields)
        ids.append(f"id_{i}_{entry['movie_name']}")

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Done! Ingested {len(documents)} scenes.")

if __name__ == "__main__":
    ingest_metadata()