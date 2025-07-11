# 📚 Book Recommendation System

A machine learning-powered book recommender system with a modern web interface built using Flask. It supports three powerful models for recommendations:

- ✅ **Popularity-Based**
- ✅ **Collaborative Filtering**
- ✅ **Content-Based Clustering (TF-IDF + K-Means)**

The project features a dark-themed UI, personalized suggestions, and optimized performance with precomputed model artifacts.

---

## 🧠 Overview

This system uses real-world datasets to deliver book recommendations through three distinct models:

- **Popularity-Based**: Recommends top-rated books with enough user votes.
- **Collaborative Filtering**: Suggests books similar to the input title using cosine similarity.
- **K-Means Clustering**: Recommends books from the same cluster based on TF-IDF vectorized metadata.

---

## 📂 Datasets

- `books.csv`: ISBN, title, author, publisher, year, image URLs
- `users.csv`: User metadata
- `ratings.csv`: User-book ratings

All datasets are stored in `/dataset`.

---

## 🧱 Project Structure

### 📊 Data Preprocessing

- Merges datasets and removes nulls/duplicates
- Creates artifacts for model deployment:
  - `popular.pkl`: Top 50 books with ≥250 ratings
  - `pt.pkl`: Pivot table of user-book ratings
  - `books.pkl`: Metadata for books
  - `similarity_scores.pkl`: Cosine similarity matrix
  - `tfidf_vectorizer.pkl`: Fitted TF-IDF model
  - `kmeans_model.pkl`: KMeans model (20 clusters)
  - `books_with_clusters.pkl`: Book metadata + cluster labels

### 🌐 Flask Web App

- **Home Page `/`**: Displays top 50 popular books with images and ratings.
- **Recommendation Page `/recommend`**: Accepts input book title + model type (Collaborative/KMeans).
- **Results Page `/recommend_books`**: Shows personalized recommendations.

---

## 🌈 CSS & UI Styling

- Dark theme with green accent (`#00a65a`)
- Glowing headings and card hovers
- Fully responsive for desktop and mobile
- Styled alerts and error messages

---

## 📊 Visualizations

- Scatter plot: Ratings vs Average Rating
- Heatmap: Cosine similarity of top 20 books
- Cluster diagram: K-Means clustering via TF-IDF

---

## ⚙️ Requirements

Install Python libraries:

```bash
pip install flask numpy pandas matplotlib seaborn scikit-learn
```

Ensure the following files exist:

- Pickles: `popular.pkl`, `pt.pkl`, `books.pkl`, `similarity_scores.pkl`, `tfidf_vectorizer.pkl`, `kmeans_model.pkl`, `books_with_clusters.pkl`
- HTML: `/templates/index.html`, `/templates/recommend.html`
- CSS: `/static/styles.css`

---

## 🛠️ Usage

### 1. Setup folders:

```bash
mkdir dataset templates static artifacts
```

Place:
- Datasets in `/dataset`
- HTML files in `/templates`
- CSS file in `/static`
- `.pkl` files in `/artifacts`

### 2. Run preprocessing:

```bash
jupyter notebook Book-Recommender.ipynb
```

### 3. Launch web app:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 💬 Example Interaction

1. Navigate to `/recommend`
2. Enter a book title (e.g., `1984`)
3. Choose a model (Collaborative or KMeans)
4. View up to 5 recommended books with title, author, and image.

> If the title isn't found, the system shows a user-friendly error message.

---

## 🚀 Future Improvements

- Add fuzzy matching for user input (typo tolerance)
- Add genres and user profiles for better personalization
- Tune number of KMeans clusters via elbow method
- Deploy to Heroku/AWS
- Add user login + history of recommendations
- Add loading animations for enhanced UX
- Cache frequent recommendations for performance

---

## 📜 License

MIT License – free to use, modify, and distribute.

---

## 👤 Author

**Pranav Prasad**  
📬 [E-mail](mailto:elixirbusiness21@gmail.com)  
🔗 [Mr-Elixir](https://github.com/Mr-Elixir)
