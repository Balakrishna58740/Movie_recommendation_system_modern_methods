import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
import json

# Initialize FastAPI app
app = FastAPI()

# MongoDB connection setup (assuming you've already set this up)
client = AsyncIOMotorClient('mongodb://localhost:27017')
mongodb = client.get_database('your_database_name')

# Download the NLTK punkt tokenizer if not already installed
nltk.download('punkt')

# Fetch data from MongoDB
@app.get("/data")
async def fetch_data():
    try:
        # Fetch all data from the 'Movies' collection
        datas = await mongodb['Movies'].find().to_list(length=100)  # limit to 100 records
        # Return the list of _id values as strings
        for data in datas:
            data['_id'] = str(data['_id'])
        return datas
    except Exception as e:
        return {"error": str(e)} 

# Preprocess text (tokenization)
def preprocess_text(text):
    return word_tokenize(text.lower())

# Function to train the Word2Vec model
def train_word2vec_model(movies_data):
    # Create a 'description' column from other relevant columns (genres and movie_rated)
    movies_data['description'] = movies_data['genres'] + " " + movies_data['movie_rated']
    
    # Preprocess descriptions (tokenization)
    movies_data['tokens'] = movies_data['description'].apply(preprocess_text)
    
    # Train Word2Vec model
    model = Word2Vec(sentences=movies_data['tokens'], vector_size=100, window=5, min_count=1, workers=4)
    model.save("movie_word2vec.model")
    return model

# Function to compute the vector representation of a movie's description
def get_movie_vector(movie_title, model, movies_data):
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
def get_similar_movies(movie_title, top_n=5, model=None, movies_data=None):
    try:
        target_vector = get_movie_vector(movie_title, model, movies_data)
    except ValueError as e:
        return str(e)
    
    similarities = []
    for _, row in movies_data.iterrows():
        other_title = row['name']
        if other_title == movie_title:
            continue  # Skip comparing the movie with itself
        
        try:
            other_vector = get_movie_vector(other_title, model, movies_data)
            similarity = cosine_similarity([target_vector], [other_vector])[0][0]
            similarities.append((other_title, similarity))
        except ValueError:
            continue  # Skip movies with no valid vectors
    
    # Sort by similarity in descending order and return the top N results
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [movie[0] for movie in sorted_similarities[:top_n]]

# Example endpoint for testing
@app.get("/similar_movies/{movie_title}")
async def fetch_similar_movies(movie_title: str):
    try:
        # Fetch data from MongoDB
        datas = await fetch_data()
        
        # Convert the data into a pandas DataFrame
        movies_data = pd.DataFrame(datas)
        
        # Train Word2Vec model
        model = train_word2vec_model(movies_data)
        
        # Get similar movies
        similar_movies = get_similar_movies(movie_title, top_n=5, model=model, movies_data=movies_data)
        return {"similar_movies": similar_movies}
    except Exception as e:
        return {"error": str(e)}