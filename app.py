import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# Library Book Recommendation System
# NLP: TF-IDF Vectorization + Cosine Similarity
# ------------------------------------------------------------

st.set_page_config(
    page_title="Library Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Step 3: Fit TF-IDF on the book catalogue.
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)
tfidf_matrix = vectorizer.fit_transform(book_text)

# ------------------------------------------------------------
# Streamlit user interface
# ------------------------------------------------------------
st.title("📚 Library Book Recommendation System")
st.write(
    "Find books that match your interests using Natural Language Processing (NLP). "
    "The system compares your query with the book catalogue using TF-IDF and cosine similarity."
)

st.subheader("Tell us what you like to read")
query = st.text_area(
    "Enter your interests:",
    value="I enjoy practical books about data science and Python.",
    placeholder="Example: I want to learn machine learning with Python...",
    height=100
)

recommend_button = st.button("Recommend Books", type="primary")

if recommend_button:
    if not query.strip():
        st.warning("Please enter your reading interests before requesting recommendations.")
    else:
        # Step 4: Convert the reader's query into a TF-IDF vector.
        query_vector = vectorizer.transform([query.strip()])

        # Step 5: Calculate cosine similarity against all book vectors.
        scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Step 6: Select the five books with the highest similarity scores.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )[:5]

        st.subheader("Top 5 Recommended Books")
        st.caption("Ranked by cosine similarity to your interests.")

        for rank, index in enumerate(ranked_indices, start=1):
            book = books[index]
            score = float(scores[index])

            with st.container(border=True):
                col_rank, col_details, col_score = st.columns([0.6, 4, 1.2])
                with col_rank:
                    st.markdown(f"### #{rank}")
                with col_details:
                    st.markdown(f"**{book['title']}**")
                    st.write(f"**Author:** {book['author']}")
                    st.write(f"**Genre:** {book['genre']}")
                    st.write(book["description"])
                with col_score:
                    st.metric("Similarity", f"{score * 100:.2f}%")

        st.info(
            "Similarity scores show how closely the catalogue text matches your query. "
            "They are not ratings of book quality."
        )

# Optional catalogue preview
with st.expander("View all 15 books in the catalogue"):
    for book in books:
        st.markdown(f"**{book['title']}** — {book['author']} ({book['genre']})")
        st.write(book["description"])
