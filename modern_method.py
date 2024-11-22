import asyncio
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from main import fetch_data
from main import app
from db_config import mongodb
# Download the NLTK punkt tokenizer if not already installed
nltk.download('punkt')


movies_data = pd.read_csv('Comedy.csv')


# Create a 'description' column from other relevant columns
# For example, concatenating 'genres' and 'movie_rated'
movies_data['description'] = movies_data['genres'] + " " + movies_data['movie_rated']

# Preprocessing: Tokenize and lowercase the movie descriptions
def preprocess_text(text):
    return word_tokenize(text.lower())

movies_data['tokens'] = movies_data['description'].apply(preprocess_text)
movies_data['tokens'].head()
# Train the Word2Vec model using tokenized movie descriptions
model = Word2Vec(sentences=movies_data['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Save the trained Word2Vec model
model.save("movie_word2vec.model")

# Function to compute the vector representation of a movie's description
def get_movie_vector(movie_title):
    # Find the row corresponding to the movie title
    movie_row = movies_data[movies_data['name'] == movie_title]
    if movie_row.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in the dataset.")
    
    # Get the tokenized description
    tokens = movie_row['tokens'].values[0]
    
    # Calculate the average vector for the tokens
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not word_vectors:
        raise ValueError(f"No valid word vectors found for movie '{movie_title}'.")
    
    movie_vector = sum(word_vectors) / len(word_vectors)
    return movie_vector

# Function to find similar movies
def get_similar_movies(movie_title, top_n=5):
    try:
        target_vector = get_movie_vector(movie_title)
    except ValueError as e:
        return str(e)
    
    similarities = []
    for _, row in movies_data.iterrows():
        other_title = row['name']
        if other_title == movie_title:
            continue  # Skip comparing the movie with itself
        
        try:
            other_vector = get_movie_vector(other_title)
            similarity = cosine_similarity([target_vector], [other_vector])[0][0]
            similarities.append((other_title, similarity))
        except ValueError:
            continue  # Skip movies with no valid vectors
    
    # Sort by similarity in descending order and return the top N results
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [movie[0] for movie in sorted_similarities[:top_n]]
count = 1
# Example usage
# movies_data['title'] = movies_data['name']
# for i in movies_data['title']:
#     print(i)
example_movie = input("Enter a movie title: ")
similar_movies = get_similar_movies(example_movie)
print(f"Movies similar to '{example_movie}': {similar_movies}")