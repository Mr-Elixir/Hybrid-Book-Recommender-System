from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load preprocessed data and models
popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))
kmeans = pickle.load(open('kmeans_model.pkl','rb'))
books_with_clusters = pickle.load(open('books_with_clusters.pkl','rb'))

def recommend_books_kmeans(book_title, n_recommendations=5):
    idxs = books_with_clusters[books_with_clusters['Book-Title'].str.lower() == book_title.lower()].index
    if len(idxs) == 0:
        return []
    book_idx = idxs[0]
    cluster_label = books_with_clusters.at[book_idx, 'cluster']
    cluster_books = books_with_clusters[books_with_clusters['cluster'] == cluster_label].copy()
    cluster_books = cluster_books.drop(book_idx, errors='ignore')
    
    tfidf_matrix = tfidf.transform(books_with_clusters['metadata'])
    book_vec = tfidf_matrix[book_idx]
    cluster_vecs = tfidf_matrix[cluster_books.index]
    
    similarities = cosine_similarity(book_vec, cluster_vecs).flatten()
    top_idxs = similarities.argsort()[::-1][:n_recommendations]
    
    recommendations = []
    for i in top_idxs:
        rb = cluster_books.iloc[i]
        recommendations.append([rb['Book-Title'], rb['Book-Author'], rb['Image-URL-M']])
    return recommendations

@app.route('/')
def home():
    top = []
    for _, r in popular_df.head(50).iterrows():
        top.append((r['Book-Title'], r['Book-Author'], r['num_ratings'], round(r['avg_rating'],2), r['Image-URL-M']))
    return render_template('index.html', books=top)

@app.route('/recommend')
def recommend_page():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend_books():
    model = request.form.get('model')
    user_input = request.form.get('user_input', '').strip()
    
    # Content-Based
    if model == 'content':
        lower_titles = [t.lower() for t in pt.index]
        if user_input.lower() not in lower_titles:
            return render_template('recommend.html', data=[], error=f"No recommendations for '{user_input}'.")
        idx = lower_titles.index(user_input.lower())
        sims = sorted(list(enumerate(similarity_scores[idx])), key=lambda x: x[1], reverse=True)[1:6]
        
        data = []
        for i, _ in sims:
            tmp = books[books['Book-Title']==pt.index[i]].drop_duplicates('Book-Title')
            if not tmp.empty:
                t = tmp.iloc[0]
                data.append([t['Book-Title'], t['Book-Author'], t['Image-URL-M']])
        return render_template('recommend.html', data=data)
    
    # KMeans Cluster-Based
    else:
        data = recommend_books_kmeans(user_input)
        if not data:
            return render_template('recommend.html', data=[], error=f"No recommendations for '{user_input}'.")
        return render_template('recommend.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
