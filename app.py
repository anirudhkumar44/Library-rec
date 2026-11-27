# Problem 9: Library Book Recommendation System
# NLP using TF-IDF and Cosine Similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------
# Step 1: Create a catalogue of 15 books
# ------------------------------------------------

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

# ------------------------------------------------
# Step 2: Combine relevant book information
# ------------------------------------------------

book_text = []

for book in books:
    text = (
        book["title"] + " " +
        book["author"] + " " +
        book["genre"] + " " +
        book["description"]
    )
    book_text.append(text)

# ------------------------------------------------
# Step 3: Convert book descriptions into TF-IDF
# vectors
# ------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(book_text)

# ------------------------------------------------
# Step 4: Accept reader interests as input
# ------------------------------------------------

query = input(
    "\nEnter your interests (e.g. I enjoy practical books about data science and Python): "
).strip()

if not query:
    query = "I enjoy practical books about data science and Python."
    print("Using sample query:", query)

# ------------------------------------------------
# Step 5: Convert reader query into TF-IDF vector
# ------------------------------------------------

query_vector = vectorizer.transform([query])

# ------------------------------------------------
# Step 6: Calculate cosine similarity
# ------------------------------------------------

similarity_scores = cosine_similarity(
    query_vector,
    tfidf_matrix
).flatten()

# ------------------------------------------------
# Step 7: Find the top 5 recommended books
# ------------------------------------------------

top_indices = similarity_scores.argsort()[::-1][:5]

# ------------------------------------------------
# Step 8: Display recommendations
# ------------------------------------------------

print("\n==========================================")
print("       TOP 5 BOOK RECOMMENDATIONS")
print("==========================================")
print("Reader's Interest:", query)

for rank, index in enumerate(top_indices, start=1):
    book = books[index]
    score = similarity_scores[index]

    print("\nRecommendation:", rank)
    print("Title:", book["title"])
    print("Author:", book["author"])
    print("Genre:", book["genre"])
    print("Similarity Score: {:.4f} ({:.2f}%)".format(
        score, score * 100
    ))

print("\n==========================================")
print("       RECOMMENDATION COMPLETED")
print("==========================================")
