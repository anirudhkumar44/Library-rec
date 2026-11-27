
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="Library Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Application Title
st.title("📚 Library Book Recommendation System")
st.write("Find books based on your interests using NLP.")

st.divider()

# Step 1: Book Catalogue

books = [
    {
        "title": "Python for Data Analysis",
        "author": "Wes McKinney",
        "genre": "Data Science",
        "description": "Practical guide to data analysis using Python, pandas and NumPy."
    },
    {
        "title": "Hands-On Machine Learning",
        "author": "Aurélien Géron",
        "genre": "Machine Learning",
        "description": "Practical machine learning techniques using Python and Scikit-learn."
    },
    {
        "title": "Data Science from Scratch",
        "author": "Joel Grus",
        "genre": "Data Science",
        "description": "Introduction to data science concepts, statistics and Python programming."
    },
    {
        "title": "Python Crash Course",
        "author": "Eric Matthes",
        "genre": "Programming",
        "description": "Beginner friendly book for learning Python programming with practical projects."
    },
    {
        "title": "Deep Learning",
        "author": "Ian Goodfellow",
        "genre": "Artificial Intelligence",
        "description": "Explains deep learning, neural networks and artificial intelligence."
    },
    {
        "title": "The Data Warehouse Toolkit",
        "author": "Ralph Kimball",
        "genre": "Data Management",
        "description": "Learn data warehousing, database design and business intelligence."
    },
    {
        "title": "Automate the Boring Stuff with Python",
        "author": "Al Sweigart",
        "genre": "Programming",
        "description": "Practical Python automation for everyday tasks and problem solving."
    },
    {
        "title": "Introduction to Statistical Learning",
        "author": "Gareth James",
        "genre": "Statistics",
        "description": "Learn statistical learning, regression, classification and data analysis."
    },
    {
        "title": "Artificial Intelligence: A Modern Approach",
        "author": "Stuart Russell",
        "genre": "Artificial Intelligence",
        "description": "Comprehensive introduction to artificial intelligence and intelligent systems."
    },
    {
        "title": "Think Python",
        "author": "Allen Downey",
        "genre": "Programming",
        "description": "Learn computer programming and problem solving using Python."
    },
    {
        "title": "Storytelling with Data",
        "author": "Cole Nussbaumer Knaflic",
        "genre": "Data Visualization",
        "description": "Practical methods for data visualization and communicating insights."
    },
    {
        "title": "R for Data Science",
        "author": "Hadley Wickham",
        "genre": "Data Science",
        "description": "Learn data science, data visualization and data transformation using R."
    },
    {
        "title": "The Pragmatic Programmer",
        "author": "David Thomas",
        "genre": "Software Development",
        "description": "Practical software development techniques and programming best practices."
    },
    {
        "title": "Pattern Recognition and Machine Learning",
        "author": "Christopher Bishop",
        "genre": "Machine Learning",
        "description": "Mathematical introduction to pattern recognition and machine learning."
    },
    {
        "title": "Data Mining: Concepts and Techniques",
        "author": "Jiawei Han",
        "genre": "Data Mining",
        "description": "Learn data mining techniques, pattern discovery and data analysis."
    }
]

# Step 2: Combine Book Information

book_text = []

for book in books:
    text = (
        book["title"] + " " +
        book["author"] + " " +
        book["genre"] + " " +
        book["description"]
    )

    book_text.append(text)

# Step 3: Convert Books into TF-IDF Vectors

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(book_text)

# Step 4: User Input

st.subheader("🔍 Find Your Favourite Books")

query = st.text_area(
    "Enter your interests:",
    placeholder="Example: I enjoy practical books about data science and Python."
)

# Number of Recommendations

number = st.selectbox(
    "Number of recommendations:",
    [3, 5, 10]
)

# Step 5: Recommendation Button

if st.button("Recommend Books", type="primary"):

    if query.strip() == "":
        st.warning("Please enter your interests first.")

    else:
        # Convert Query into TF-IDF Vector

        query_vector = vectorizer.transform([query])

        # Step 6: Calculate Cosine Similarity

        similarity_scores = cosine_similarity(
            query_vector,
            tfidf_matrix
        )[0]

        # Step 7: Get Top Recommended Books

        top_indices = similarity_scores.argsort()[::-1][:number]

        st.divider()

        st.subheader("📖 Recommended Books for You")

        # Step 8: Display Results

        found = False

        for rank, index in enumerate(top_indices, start=1):

            book = books[index]
            score = similarity_scores[index]

            if score > 0:
                found = True

                with st.container(border=True):

                    st.markdown(
                        f"### {rank}. {book['title']}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Author:**", book["author"])
                        st.write("**Genre:**", book["genre"])

                    with col2:
                        st.metric(
                            "Similarity Score",
                            f"{score:.4f}"
                        )

                    st.write("**Description:**")
                    st.write(book["description"])

            else:
                continue

        if not found:
            st.info(
                "No matching books found. "
                "Try entering different interests."
            )

        st.success("Book recommendations generated successfully!")

# Step 9: Display Complete Catalogue

st.divider()

with st.expander("📚 View All Available Books"):

    for book in books:

        st.markdown(f"**{book['title']}**")
        st.write("Author:", book["author"])
        st.write("Genre:", book["genre"])
        st.write("Description:", book["description"])
        st.divider()

# Footer

st.caption(
    "Library Book Recommendation System | "
    "Powered by Python, Streamlit and NLP"
)
