# ===============================================================
# DASHBOARD STREAMLIT - ANALYSE & COMPARAISON DES MODÈLES
# ===============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch, time

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="Dashboard Produits Capillaires", layout="wide")

# --------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# --------------------------------------------------------------
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    hair_categories = [
        'Hair', 'Conditioner', 'Dry Shampoo', 'Shampoo', 'Hair Masks', 'Hair Oil',
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
# BASELINE (TF-IDF)
# --------------------------------------------------------------
@st.cache_resource
def get_baseline(corpus):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

tfidf_vectorizer, tfidf_matrix = get_baseline(df_hair['corpus'])

def recommend_baseline(query, top_n=5):
    q_vec = tfidf_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx = sims.argsort()[:-top_n-1:-1]
    return df_hair.iloc[idx]

# --------------------------------------------------------------
# MODÈLE AVANCÉ (SBERT)
# --------------------------------------------------------------
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
# ANALYSE EXPLORATOIRE
# --------------------------------------------------------------
st.title("📊 Dashboard - Produits Capillaires")

st.header("Analyse Exploratoire")
st.subheader("Top 20 Marques")
fig1 = px.bar(df_hair['brand'].value_counts().nlargest(20).reset_index(),
              x='brand', y='count')
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Distribution des prix")
fig2 = px.histogram(df_hair, x='price', nbins=50)
st.plotly_chart(fig2, use_container_width=True)

# Graphique 3 : Distribution du prix par catégorie (boxplot)

st.subheader("Distribution des Prix par Catégorie")

# Create the box plot
fig = px.box(
    df_hair,
    x="category",
    y="price",
    color="category",
    labels={"category": "Catégorie", "price": "Prix ($)"}
)

# Update layout
fig.update_layout(
    xaxis_title="Catégorie",
    yaxis_title="Prix ($)",
    showlegend=False,
    title="Distribution des Prix par Catégorie"
)

# Display the figure
st.plotly_chart(fig, use_container_width=True)

st.subheader("WordCloud des ingrédients")
text_ing = " ".join(df_hair['ingredients'].dropna())
stopwords = set(STOPWORDS).union({"water","aqua","eau","citric","acid","fragrance"})
wc = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text_ing)
plt.figure(figsize=(10,5)); plt.imshow(wc, interpolation="bilinear")
plt.axis("off"); st.pyplot(plt)

# --------------------------------------------------------------
# COMPARAISON DES MODÈLES
# --------------------------------------------------------------
st.header("Comparaison des modèles")

test_suite = [
    {"query":"shampoo for oily hair","category_keyword":"Shampoo","corpus_keyword":"oily"},
    {"query":"dry mask for dry hair","category_keyword":"Mask","corpus_keyword":"dry"},
    {"query":"curly hair conditioner","category_keyword":"Conditioner","corpus_keyword":"curly"},
    {"query":"light hair oil for shine","category_keyword":"Oil","corpus_keyword":"shine"},
    {"query":"leave-in conditioner for frizz","category_keyword":"Leave-In","corpus_keyword":"frizz"}
]

def calc_metrics(recs, ground_truth_idx, top_n):
    rec_idx = set(recs.index); gt = set(ground_truth_idx)
    tp = len(rec_idx & gt)
    prec = tp/top_n if top_n>0 else 0
    rec = tp/len(gt) if len(gt)>0 else 0
    f1 = 2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0
    return prec,rec,f1

def evaluate_model_with_time(model_func, test_suite, top_n=10):
    metrics = []; total_time = 0
    for test in test_suite:
        gt = df_hair[df_hair['category'].str.contains(test['category_keyword'],case=False) &
                     df_hair['corpus'].str.contains(test['corpus_keyword'],case=False)]
        if gt.empty: continue
        start = time.time()
        recs = model_func(test['query'], top_n=top_n)
        elapsed = time.time() - start; total_time += elapsed
        p,r,f1 = calc_metrics(recs, gt.index, top_n)
        metrics.append({"query": test['query'], "precision":p, "recall":r, "f1_score":f1})
    avg = pd.DataFrame(metrics).mean(numeric_only=True)
    avg["time_sec"] = total_time/len(metrics)
    return avg

baseline_avg = evaluate_model_with_time(recommend_baseline, test_suite)
advanced_avg = evaluate_model_with_time(recommend_advanced, test_suite)

comparison_df = pd.DataFrame({
    "Baseline (TF-IDF)": baseline_avg,
    "Avancé (SBERT)": advanced_avg
}).T

st.subheader("Tableau comparatif")
st.dataframe(comparison_df)

fig3 = px.bar(comparison_df.reset_index(), x="index",
              y=["precision","recall","f1_score"],
              barmode="group", title="Précision / Rappel / F1")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.bar(comparison_df.reset_index(), x="index",
              y=["time_sec"], title="Temps moyen d'inférence (s)")
st.plotly_chart(fig4, use_container_width=True)

best = "Avancé (SBERT)" if advanced_avg["f1_score"] > baseline_avg["f1_score"] else "Baseline (TF-IDF)"
st.success(f"🏆 Modèle le plus performant : **{best}**")
