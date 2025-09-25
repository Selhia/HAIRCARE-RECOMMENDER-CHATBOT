# ===============================================================
# CHATBOT STREAMLIT - RECOMMANDATION DE PRODUITS CAPILLAIRES
# ===============================================================

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="Chatbot Produits Capillaires", layout="centered")

# --------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# --------------------------------------------------------------
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    hair_categories = [
        'Hair', 'Conditioner', 'Dry Shampoo', 'Hair Masks', 'Hair Oil',
        'Hair Primers', 'Hair products', 'Hair Spray', 'Leave-In Conditioner'
    ]
    df = df[df['category'].isin(hair_categories)].copy()
    for col in ['name','brand','details','ingredients']:
        df[col] = df[col].fillna('')
    df['corpus'] = (
        "Catégorie: " + df['category'] + ". " +
        "Nom: " + df['name'] + ". " +
        "Marque: " + df['brand'] + ". " +
        "Description: " + df['details'] + ". " +
        "Ingrédients: " + df['ingredients']
    )
    return df

df_hair = load_data("data/sephora_website_dataset.csv")

# --------------------------------------------------------------
# BASELINE & AVANCÉ
# --------------------------------------------------------------
@st.cache_resource
def get_baseline(corpus):
    v = TfidfVectorizer(stop_words='english', max_features=5000)
    m = v.fit_transform(corpus)
    return v, m

tfidf_vectorizer, tfidf_matrix = get_baseline(df_hair['corpus'])

def recommend_baseline(query, top_n=5):
    q_vec = tfidf_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx = sims.argsort()[:-top_n-1:-1]
    return df_hair.iloc[idx]

@st.cache_resource
def get_advanced(corpus):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus.tolist(), convert_to_tensor=True)
    return model, embeddings

advanced_model, product_embeddings = get_advanced(df_hair['corpus'])

def recommend_advanced(query, top_n=5):
    q_emb = advanced_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, product_embeddings, top_k=top_n)[0]
    idx = [h['corpus_id'] for h in hits]
    return df_hair.iloc[idx]

# --------------------------------------------------------------
# LOGIQUE SMART (gamme / catégorie)
# --------------------------------------------------------------
RANGE_KEYWORDS = ['gamme','ensemble','routine','set','complet']
CATEGORY_MAPPING = {
    'shampoo':'Shampoo','shampoing':'Shampoo',
    'conditioner':'Conditioner','après-shampoing':'Conditioner',
    'mask':'Hair Masks','masque':'Hair Masks',
    'oil':'Hair Oil','huile':'Hair Oil',
    'spray':'Hair Spray',
    'leave-in':'Leave-In Conditioner','sans rinçage':'Leave-In Conditioner',
    'crème':'Leave-In Conditioner','creme':'Leave-In Conditioner'
}

def smart_recommendations(query, model_func=recommend_advanced):
    q = query.lower()
    if any(k in q for k in RANGE_KEYWORDS):
        candidates = model_func(query, top_n=50)
        routine = {'Shampoo':'Shampoo','Conditioner':'Conditioner',
                   'Mask':'Hair Masks',
                   'Treatment':['Hair Oil','Leave-In Conditioner','Hair Spray']}
        gamme, seen = [], set()
        for cat in routine.values():
            cats = [cat] if isinstance(cat,str) else cat
            for _,p in candidates.iterrows():
                if p['category'] in cats and p['id'] not in seen:
                    gamme.append(p); seen.add(p['id']); break
        return pd.DataFrame(gamme)
    for kw,cat in CATEGORY_MAPPING.items():
        if kw in q:
            candidates = model_func(query, top_n=50)
            return candidates[candidates['category']==cat].head(4)
    return model_func(query, top_n=4)

# --------------------------------------------------------------
# INTERFACE CHATBOT
# --------------------------------------------------------------
st.title("🤖 Chatbot Produits Capillaires")

user_query = st.chat_input("Décris ton besoin capillaire...")
if user_query:
    recs = smart_recommendations(user_query, recommend_advanced)
    st.chat_message("user").write(user_query)
    st.chat_message("assistant").write("Voici mes recommandations :")
    for _,row in recs.iterrows():
        st.markdown(f"- **{row['name']}** (*{row['brand']}*, {row['category']}) – 💲{row['price']}, ⭐ {row['rating']}")
