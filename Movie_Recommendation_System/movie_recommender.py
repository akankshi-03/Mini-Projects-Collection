# Movie Recommendation System using content-based filtering
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
movies = pd.DataFrame({
    'title': ['Inception', 'The Matrix', 'Interstellar', 'The Notebook', 'John Wick'],
    'description': [
        'dreams within dreams, sci-fi thriller',
        'virtual reality and science fiction',
        'space travel and emotional storytelling',
        'romantic love story and drama',
        'action and revenge mission'
    ]
})

# Convert descriptions to vectors
vectorizer = TfidfVectorizer()
desc_vectors = vectorizer.fit_transform(movies['description'])

# Similarity matrix
similarity_matrix = cosine_similarity(desc_vectors)

# Recommendation function
def recommend(movie_title):
    index = movies[movies['title'] == movie_title].index[0]
    similar_scores = list(enumerate(similarity_matrix[index]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)[1:3]
    recommendations = [movies.iloc[i[0]]['title'] for i in similar_scores]
    print("Because you liked '{}', you might also like:".format(movie_title))
    for rec in recommendations:
        print("- " + rec)

# Example usage
recommend("Inception")
