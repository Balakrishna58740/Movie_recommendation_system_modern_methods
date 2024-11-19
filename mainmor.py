import pandas as pd
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Load the MovieLens dataset
# For the example, we will assume the dataset has columns: movieId, title, genres, and description
movies_data= pd.read_csv('/Music.csv')

# Example data structure
# movieId | title         | genres        | description
# 1       | Toy Story     | Animation,Comedy | A story of a young boy's toys that come to life when he is away.

# Check the structure of the dataset
# Function to preprocess text (tokenizing and cleaning)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase the text
    return tokens

# Apply the preprocessing function to the movie descriptions
movies_data['tokens'] = movies_data['description'].apply(preprocess_text)
# Train the Word2Vec model using movie descriptions
model = Word2Vec(sentences=movies_data['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Save the model for later use
model.save("movie_word2vec.model")
# Function to get the vector representation of a movie's description
def get_movie_vector(movie_title):
    # Get the movie's tokenized description
    tokens = movies_data[movies_data['title'] == movie_title]['tokens'].values[0]
    
    # Get the vector for each word in the description and average them
    movie_vector = sum([model.wv[word] for word in tokens if word in model.wv]) / len(tokens)
    
    return movie_vector

# Function to get similar movies
def get_similar_movies(movie_title, top_n=5):
    movie_vector = get_movie_vector(movie_title)
    
    # Calculate cosine similarity between the target movie and all other movies
    similarities = []
    for i, row in movies_data.iterrows():
        other_movie_vector = get_movie_vector(row['title'])
        similarity = cosine_similarity([movie_vector], [other_movie_vector])
        similarities.append((row['title'], similarity[0][0]))
    
    # Sort the movies based on similarity scores and return the top N
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [movie[0] for movie in sorted_similarities[1:top_n+1]]

# Example: Get similar movies to "Toy Story"
similar_movies = get_similar_movies("Toy Story")
print(f"Movies similar to 'Toy Story': {similar_movies}")


