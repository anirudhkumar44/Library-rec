# 📚 Library Book Recommendation System

## 📌 Project Overview

The Library Book Recommendation System is a Python-based project that recommends books based on a reader's interests.

The system uses Natural Language Processing (NLP) techniques, specifically **TF-IDF (Term Frequency–Inverse Document Frequency)** and **Cosine Similarity**, to compare the reader's query with book descriptions.

Based on the similarity scores, the system recommends the top 5 most relevant books.

## 🎯 Objectives

- Create a catalogue containing at least 15 books.
- Store book title, author, genre, and description.
- Combine relevant descriptive fields into searchable text.
- Convert book descriptions into TF-IDF vectors.
- Convert the reader's query into a TF-IDF vector.
- Calculate cosine similarity between the query and books.
- Recommend the top 5 matching books.
- Display title, genre, author, and similarity score.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity

## 📂 Project Structure

```text
Library-Book-Recommendation/
│
├── book_recommendation.py
├── requirements.txt
└── README.md
```

## 📖 Book Catalogue

The project contains 15 books from different categories:

| No. | Book Title | Genre |
|---|---|---|
| 1 | Python Crash Course | Programming |
| 2 | Hands-On Machine Learning | Machine Learning |
| 3 | Data Science from Scratch | Data Science |
| 4 | Automate the Boring Stuff with Python | Programming |
| 5 | The Data Warehouse Toolkit | Data Analytics |
| 6 | Introduction to Statistical Learning | Statistics |
| 7 | Deep Learning | Deep Learning |
| 8 | Clean Code | Software Development |
| 9 | Think Python | Programming |
| 10 | Practical Statistics for Data Scientists | Data Science |
| 11 | Artificial Intelligence: A Modern Approach | Artificial Intelligence |
| 12 | The Pragmatic Programmer | Software Development |
| 13 | Data Mining: Concepts and Techniques | Data Mining |
| 14 | Python for Data Analysis | Data Science |
| 15 | The Hundred-Page Machine Learning Book | Machine Learning |

Each book also contains its author's name and a descriptive summary used by the recommendation algorithm.

## ⚙️ How the System Works

### 1. Book Catalogue

A collection of 15 books is created using Python dictionaries containing:
- Title
- Author
- Genre
- Description

### 2. Text Combination

The title, author, genre, and description of each book are combined into a single searchable text field.

### 3. TF-IDF Vectorization

TF-IDF converts the textual information into numerical vectors based on the importance of words.

### 4. User Input

The reader enters their interests or preferred topics.

Example:

```text
I enjoy practical books about data science and Python.
```

### 5. Cosine Similarity

Cosine similarity compares the reader's query vector with the TF-IDF vectors of all books.

### 6. Top 5 Recommendations

The system sorts books according to their similarity scores and displays the five most relevant recommendations.

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Library-Book-Recommendation.git
```

### Step 2: Open the Project Folder

```bash
cd Library-Book-Recommendation
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python book_recommendation.py
```

## 📦 Requirements

The following libraries are required:

```text
pandas
scikit-learn
```

## 🧪 Sample Input

```text
I enjoy practical books about data science and Python.
```

## 📊 Expected Output

The system displays the top 5 recommended books, including:

- Book title
- Author
- Genre
- Similarity score

Example recommendations may include:

1. Python for Data Analysis
2. Data Science from Scratch
3. Python Crash Course
4. Hands-On Machine Learning
5. Practical Statistics for Data Scientists

*The actual ranking and similarity percentages are calculated by the program and may differ.*

## ✨ Features

- Catalogue of 15 books.
- User-based interest searching.
- TF-IDF text vectorization.
- Cosine similarity calculation.
- Automatic top 5 book recommendations.
- Simple command-line interface.
- Beginner-friendly Python implementation.

## 🔮 Future Enhancements

- Add more books to the catalogue.
- Develop a graphical user interface using Streamlit.
- Add book ratings and reviews.
- Allow users to search by author or genre.
- Store book information in a database.
- Create personalized recommendations based on reading history.

## 📌 Conclusion

The Library Book Recommendation System demonstrates how Machine Learning and Natural Language Processing can be used to recommend books based on textual similarity.

By combining TF-IDF vectorization and cosine similarity, the system identifies books that match a reader's interests and provides the five most relevant recommendations.

This project provides a practical introduction to content-based recommendation systems using Python.

## 👨‍💻 Project Information

**Project Name:** Library Book Recommendation System

**Problem Number:** 9

**Difficulty Level:** Beginner–Intermediate

**Domain:** Machine Learning / Natural Language Processing

**Programming Language:** Python

---

⭐ If you find this project useful, consider giving the repository a star!
