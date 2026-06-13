import streamlit as st
import os
import json
import chromadb
from google import genai
from sentence_transformers import CrossEncoder
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import time

load_dotenv()

# --- RESOURCES ---
@st.cache_resource
def load_all():
    db_path = os.path.join(os.getcwd(), "..", "vector_db")
    client = chromadb.PersistentClient(path=db_path)
    # Local embedding model - does not use API quota
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    collection = client.get_collection(name="movie_scenes", embedding_function=emb_func)
    
    re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return collection, re_ranker, genai_client, emb_func

collection, re_ranker, genai_client, emb_func = load_all()

# --- LOGIC WITH QUOTA PROTECTION ---
@st.cache_data(show_spinner=False)
def get_filter_cached(query):
    """
    Caches the API response for 24 hours so repeat searches 
    don't drain your daily quota.
    """
    try:
        # Note: Using gemini-3-flash as per your latest error log
        prompt = f"Identify if any of these movies are mentioned: 'maghaheera', 'pushpa'. If found, return ONLY the name in lowercase. If none, return 'none'. Query: {query}"
        time.sleep(1) 
        response = genai_client.models.generate_content(model="gemini-3-flash", contents=prompt)
        res = response.text.strip().lower()
        return None if "none" in res else res
    except Exception as e:
        st.warning("API Quota reached or Error. Using local keyword matching.")
        query_lower = query.lower()
        if "pushpa" in query_lower: return "pushpa"
        if "maghaheera" in query_lower: return "maghaheera"
        return None

# --- UI ---
st.set_page_config(page_title="Movie Search", layout="wide")
st.title("🎬 Movie Scene Search")

st.sidebar.info("Free Tier Quota: 20 requests/day. Caching is enabled to save credits.")

query = st.text_input("What scene are you looking for?", placeholder="e.g. interesting scenes in pushpa")

if query:
    # 1. Extraction & Filter Preparation
    movie_name = get_filter_cached(query)
    where_filter = {"movie_name": movie_name} if movie_name else None

    # --- PAYLOAD & VECTOR DEBUGGING ---
    search_payload = {
        "query_texts": [query],
        "n_results": 10,
        "where_filter": where_filter,
        "include": ["documents", "metadatas", "distances"]
    }
    
    # Print to Terminal for backend logs
    print("\n--- CHROMA SEARCH PAYLOAD ---")
    print(json.dumps(search_payload, indent=2))
    
    # UI Debug Expander
    with st.expander("🛠️ Internal Search Mechanics & Payload"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Search Payload (JSON):**")
            st.json(search_payload)
        with col2:
            # Manually generate embedding to show the vector
            query_vector = emb_func([query])[0]
            st.write(f"**Vector Embedding (First 10 of {len(query_vector)} dims):**")
            st.code(query_vector[:10])

    # 2. Semantic Search Execution
    results = collection.query(
        query_texts=search_payload["query_texts"], 
        n_results=search_payload["n_results"], 
        where=search_payload["where_filter"],
        include=search_payload["include"]
    )

    if results['documents'] and len(results['documents'][0]) > 0:
        # 3. Re-ranking (Local CPU)
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        pairs = [[query, doc] for doc in docs]
        scores = re_ranker.predict(pairs)
        
        # Merge scores and metadata for final display
        final = sorted(zip(scores, metas), key=lambda x: x[0], reverse=True)[:5]

        # 4. Display Results in smaller size
        st.subheader(f"Top results for: {query}")
        for score, meta in final:
            with st.container():
                st.markdown(f"<h3 style='text-align: center;'>{meta['movie_name'].upper()}</h3>", unsafe_allow_html=True)
                
                # Using 1:2:1 ratio to make the video smaller (centered)
                left_co, cent_co, last_co = st.columns([1, 2, 1])
                with cent_co:
                    if os.path.exists(meta['video_path']):
                        st.video(meta['video_path'], start_time=int(meta['start_time']))
                    else:
                        st.error(f"Video file not found at: {meta['video_path']}")
                    
                    st.write(f"**Re-rank Score:** `{score:.4f}`")
                    st.write(f"**Scene Timing:** {meta['start_time']}s - {meta['end_time']}s")
                st.divider()
    else:
        st.error("No matching scenes found in the vector database.")

# --- FOOTER ---
if query:
    st.sidebar.write("---")
    st.sidebar.write("**Current Filter:**", movie_name if movie_name else "None")