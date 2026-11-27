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

# Step 1: Book catalogue (15 books)
books = [
    {
        "title": "Hands-On Machine Learning",
        "author": "Aurélien Géron",
        "genre": "Machine Learning",
        "description": "Practical machine learning techniques using Python and Scikit-learn."
    },
    {
        "title": "Python Crash Course",
        "author": "Eric Matthes",
        "genre": "Python Programming",
        "description": "A beginner-friendly guide to learning Python through practical projects."
    },
    {
        "title": "Deep Learning",
        "author": "Ian Goodfellow, Yoshua Bengio, and Aaron Courville",
        "genre": "Deep Learning",
        "description": "Explains deep learning concepts, neural networks, and artificial intelligence."
    },
    {
        "title": "Artificial Intelligence: A Modern Approach",
        "author": "Stuart Russell and Peter Norvig",
        "genre": "Artificial Intelligence",
        "description": "An introduction to artificial intelligence, algorithms, and intelligent systems."
    },
    {
        "title": "Data Science from Scratch",
        "author": "Joel Grus",
        "genre": "Data Science",
        "description": "Learn data science concepts, statistics, and practical implementations using Python."
    },
    {
        "title": "Introduction to Machine Learning with Python",
        "author": "Andreas C. Müller and Sarah Guido",
        "genre": "Machine Learning",
        "description": "Practical introduction to machine learning using Python and Scikit-learn."
    },
    {
        "title": "Think Python",
        "author": "Allen B. Downey",
        "genre": "Python Programming",
        "description": "Learn Python programming, problem solving, and computational thinking."
    },
    {
        "title": "The Hundred-Page Machine Learning Book",
        "author": "Andriy Burkov",
        "genre": "Machine Learning",
        "description": "Explains machine learning algorithms, models, and important concepts."
    },
    {
        "title": "Deep Learning with Python",
        "author": "François Chollet",
        "genre": "Deep Learning",
        "description": "Build neural networks and deep learning applications using Python and Keras."
    },
    {
        "title": "Practical Statistics for Data Scientists",
        "author": "Peter Bruce, Andrew Bruce, and Peter Gedeck",
        "genre": "Data Science",
        "description": "Learn statistical methods, data analysis, and practical data science techniques."
    },
    {
        "title": "Automate the Boring Stuff with Python",
        "author": "Al Sweigart",
        "genre": "Python Programming",
        "description": "Use Python programming to automate everyday tasks and improve productivity."
    },
    {
        "title": "Machine Learning Yearning",
        "author": "Andrew Ng",
        "genre": "Machine Learning",
        "description": "Practical guidance for designing, developing, and improving machine learning projects."
    },
    {
        "title": "The Data Warehouse Toolkit",
        "author": "Ralph Kimball and Margy Ross",
        "genre": "Data Analytics",
        "description": "Learn data warehouse design, data modelling, and business analytics."
    },
    {
        "title": "Natural Language Processing with Python",
        "author": "Steven Bird, Ewan Klein, and Edward Loper",
        "genre": "Natural Language Processing",
        "description": "Learn text processing, language analysis, and NLP techniques using Python."
    },
    {
        "title": "Pattern Recognition and Machine Learning",
        "author": "Christopher M. Bishop",
        "genre": "Machine Learning",
        "description": "Explore statistical methods, pattern recognition, and machine learning algorithms."
    }
]

# Step 2: Combine title, author, genre, and description into searchable text.
book_text = [
    f"{book['title']} {book['author']} {book['genre']} {book['description']}"
    for book in books
]

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
