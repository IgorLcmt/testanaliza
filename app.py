
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import io
import openai
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
import time
import json
import logging

# --- Constants ---
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 100
EXCEL_PATH = "app_data/Database.xlsx"
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Logger Setup ---
logging.basicConfig(filename='app_debug.log', level=logging.INFO)

# --- Utility Functions ---
def cache_embedding(key, embedding):
    with open(os.path.join(CACHE_DIR, f"{key}.json"), "w") as f:
        json.dump(embedding, f)

def load_cached_embedding(key):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def sanitize_domain(domain):
    return domain.replace("http://", "").replace("https://", "").split("/")[0]

# --- Scraping Function ---
def scrape_text(domain):
    domain = sanitize_domain(domain)
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(f"https://{domain}", headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        logging.warning(f"Primary scrape failed for {domain}: {e}")
    try:
        archive_url = f"http://web.archive.org/web/*/{domain}"
        res = requests.get(archive_url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        logging.error(f"Archive scrape failed for {domain}: {e}")
    return ""

# --- Embedding Function ---
def get_embeddings(texts, api_key):
    openai.api_key = api_key
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        try:
            response = openai.Embedding.create(input=batch, model=EMBEDDING_MODEL)
            batch_embeddings = [r["embedding"] for r in response["data"]]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logging.error(f"Embedding batch {i // BATCH_SIZE} failed: {e}")
            raise
        time.sleep(1)
    return embeddings

# --- Embed the Database ---
def embed_database(df, api_key):
    texts, ids = [], []
    for idx, row in df.iterrows():
        record_id = row['MI Transaction ID']
        cached = load_cached_embedding(record_id)
        if cached:
            df.at[idx, "embedding"] = cached
        else:
            website_text = scrape_text(row.get("Web page", ""))
            composite = " ".join(filter(None, [
                str(row["Business Description"]),
                str(row["Primary Industry"]),
                website_text
            ]))
            texts.append(composite)
            ids.append(record_id)
            df.at[idx, "Website Text"] = website_text
            df.at[idx, "Composite"] = composite

    if texts:
        new_embeddings = get_embeddings(texts, api_key)
        for rid, emb in zip(ids, new_embeddings):
            cache_embedding(rid, emb)
            df.loc[df['MI Transaction ID'] == rid, "embedding"] = [emb]
    return df

# --- Top Matches Finder ---
def find_top_matches(df, query, api_key, top_n=10):
    query_embedding = get_embeddings([query], api_key)[0]
    emb_matrix = np.vstack(df["embedding"].values)
    emb_matrix_norm = normalize(emb_matrix)
    query_norm = normalize(np.array(query_embedding).reshape(1, -1))
    similarities = cosine_similarity(query_norm, emb_matrix_norm)[0]
    df["Similarity Score"] = similarities
    top = df.sort_values(by="Similarity Score", ascending=False).head(top_n).copy()
    top["Reason for Match"] = "High semantic + content + industry similarity"
    return top[[
        'Target/Issuer Name', 'MI Transaction ID', 'Implied Enterprise Value/ EBITDA (x)',
        'Business Description', 'Primary Industry', 'Web page', 'Similarity Score', 'Reason for Match'
    ]]

# --- Streamlit UI ---
st.set_page_config(page_title="CMT analiza mnożników pod wycene 🔍", layout="wide")
st.title("CMT analiza mnożników pod wycene 🔍")

@st.cache_data
def load_database():
    df = pd.read_excel(EXCEL_PATH)
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={
        'Business Description
(Target/Issuer)': 'Business Description',
        'Primary Industry
(Target/Issuer)': 'Primary Industry'
    })
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna(subset=[
        'Target/Issuer Name', 'MI Transaction ID', 'Implied Enterprise Value/ EBITDA (x)',
        'Business Description', 'Primary Industry'
    ])
    return df

api_key = st.secrets["openai"]["api_key"]
query_input = st.sidebar.text_area("✏️ Paste company profile here:", height=200)

if api_key and query_input:
    try:
        df = load_database()
        with st.spinner("Embedding and scraping in progress..."):
            df_prepared = embed_database(df, api_key)
        results = find_top_matches(df_prepared, query_input, api_key)
        st.success("Top matches found:")
        st.dataframe(results, use_container_width=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results.to_excel(writer, index=False, sheet_name="Top Matches")
        st.download_button("📥 Download Excel", data=output.getvalue(),
                           file_name="Top_Matches.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info(" Enter company profile to begin.")
