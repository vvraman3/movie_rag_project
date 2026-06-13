import streamlit as st
import os
import json
import chromadb
from google import genai
from sentence_transformers import CrossEncoder
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# --- RESOURCES ---
@st.cache_resource
def load_all():
    db_path = os.path.join(os.getcwd(), "..", "vector_db")
    client = chromadb.PersistentClient(path=db_path)
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    collection = client.get_collection(name="movie_scenes", embedding_function=emb_func)
    
    re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return collection, re_ranker, genai_client

collection, re_ranker, genai_client = load_all()

# --- LOGIC ---
def get_filter(query):
    prompt = f"Identify if any of these movies are mentioned: 'maghaheera', 'pushpa'. If found, return ONLY the name in lowercase. If none, return 'none'. Query: {query}"
    response = genai_client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
    res = response.text.strip().lower()
    return None if "none" in res else res

# --- UI ---
st.set_page_config(page_title="Movie Search", layout="wide") # Set to wide for better column control
st.title("🎬 Movie Scene Search")
query = st.text_input("What scene are you looking for?", placeholder="e.g. funny scenes in pushpa")

if query:
    movie_name = get_filter(query)
    
    # Semantic Search
    where_filter = {"movie_name": movie_name} if movie_name else None
    results = collection.query(
        query_texts=[query], 
        n_results=10, 
        where=where_filter,
        include=["documents", "metadatas"]
    )

    if results['documents']:
        # Re-ranking
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        pairs = [[query, doc] for doc in docs]
        scores = re_ranker.predict(pairs)
        
        final = sorted(zip(scores, metas), key=lambda x: x[0], reverse=True)[:5]

        # --- UPDATED DISPLAY LOGIC ---
        for score, meta in final:
            with st.container():
                # Center the header
                st.markdown(f"<h3 style='text-align: center;'>{meta['movie_name'].upper()}</h3>", unsafe_allow_index=True)
                
                # Create columns to shrink the video width (1:2:1 ratio centers it and makes it smaller)
                left_co, cent_co, last_co = st.columns([1, 2, 1])
                
                with cent_co:
                    if os.path.exists(meta['video_path']):
                        # Height is automatically adjusted by Streamlit based on width
                        st.video(meta['video_path'], start_time=int(meta['start_time']))
                    else:
                        st.error(f"Video file not found at: {meta['video_path']}")
                    
                    st.write(f"**Scene Timing:** {meta['start_time']}s - {meta['end_time']}s")
                
                st.divider()