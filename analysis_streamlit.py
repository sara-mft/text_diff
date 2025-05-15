import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# ---------- Styling ----------
st.set_page_config(page_title="NLU Data Analysis", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
#local_css("assets/style.css")

# ---------- Upload ----------
st.sidebar.title("Data Upload")
uploaded_file = st.sidebar.file_uploader("Excel File", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if "query" not in df.columns or "class" not in df.columns:
        st.error("Your file must contain 'query' and 'class' columns.")
        st.stop()
    
    df.dropna(subset=["query", "class"], inplace=True)
    df["char_count"] = df["query"].apply(len)
    df["word_count"] = df["query"].apply(lambda x: len(x.split()))

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs(["Stats", "N-grams", "Keywords", "t-SNE"])

    # ---------- Tab 1: Stats ----------
    with tab1:
        st.subheader("Descriptive Statistics by Class")
        stats = df.groupby("class")[["char_count", "word_count"]].mean().round(2).reset_index()
        st.dataframe(stats)

        fig1 = px.bar(stats, x='class', y='word_count', title="Average Word Count per Class",
                      color='class', height=400)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(df, x="class", y="char_count", title="Character Count Distribution", color="class")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- Tab 2: N-grams ----------
    with tab2:
        st.subheader("Top N-grams per Class")
        ngram_n = st.slider("N for N-gram", 1, 3, 2)
        top_k = st.slider("Top N-grams", 5, 20, 10)
        class_choice = st.selectbox("Select class", df["class"].unique())

        def get_ngrams(corpus, ngram_range=(1, 1), top_n=10):
            vec = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(corpus)
            bag = vec.transform(corpus)
            sum_words = bag.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]

        ngrams = get_ngrams(df[df["class"] == class_choice]["query"], 
                            ngram_range=(ngram_n, ngram_n), top_n=top_k)
        ng_df = pd.DataFrame(ngrams, columns=["N-gram", "Frequency"])
        st.dataframe(ng_df)

        fig3 = px.bar(ng_df, x="N-gram", y="Frequency", color="Frequency", title="Top N-grams")
        st.plotly_chart(fig3, use_container_width=True)

    # ---------- Tab 3: Keywords ----------
    with tab3:
        st.subheader("Top TF-IDF Keywords by Class")

        def extract_keywords(df, top_n=10):
            result = {}
            for label in df["class"].unique():
                corpus = df[df["class"] == label]["query"]
                tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
                mat = tfidf.fit_transform(corpus)
                scores = mat.mean(axis=0).A1
                top = scores.argsort()[-top_n:][::-1]
                words = [tfidf.get_feature_names_out()[i] for i in top]
                result[label] = words
            return result

        tfidf_words = extract_keywords(df)
        cols = st.columns(len(tfidf_words))
        for idx, (cls, words) in enumerate(tfidf_words.items()):
            with cols[idx]:
                st.markdown(f"**{cls}**")
                st.write(", ".join(words))

        # WordCloud per class
        st.subheader("Word Cloud")
        wc_class = st.selectbox("Class for WordCloud", df["class"].unique())

        wc_text = " ".join(df[df["class"] == wc_class]["query"])
        wc = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate(wc_text)

        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

    # ---------- Tab 4: t-SNE ----------
    with tab4:
        st.subheader("t-SNE ")

        from sentence_transformers import SentenceTransformer

        with st.spinner("Loading multilingual model and computing embeddings..."):
            model = SentenceTransformer("intfloat/multilingual-e5-base")

            # HuggingFace E5 expects "query: <text>" for query embeddings
            sentences = ["query: " + q for q in df["query"].astype(str).tolist()]
            embeddings = model.encode(sentences, show_progress_bar=True)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_emb = tsne.fit_transform(embeddings)

        df["tsne_x"] = X_emb[:, 0]
        df["tsne_y"] = X_emb[:, 1]

        fig_tsne = px.scatter(
            df, x="tsne_x", y="tsne_y", color="class",
            hover_data=["query"], title="t-SNE",
            width=1000, height=600
        )
        st.plotly_chart(fig_tsne, use_container_width=True)
    st.info("Upload your Excel file to get started.")
