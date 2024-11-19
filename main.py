import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommendationModel:
    def __init__(self, data):
        """
        Initialize the recommendation model with movie data.
        """
        self.data = data
        self.cosine_sim = None
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.fit_model()
    
    def preprocess_data(self):
        """
        Preprocess the movie dataset by combining features like genres, year, and rating
        into a single 'features' column.
        """
        self.data['features'] = self.data['genres'] + ' ' + self.data['year'].astype(str) + ' ' + self.data['movie_rated']
    
    def fit_model(self):
        """
        Train the recommendation model using TF-IDF vectorization and cosine similarity.
        """
        self.preprocess_data()
        # Vectorizing the 'features' column with TF-IDF Vectorizer
        tfidf_matrix = self.tfidf.fit_transform(self.data['features'])
        
        # Compute cosine similarity between movies
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    def get_recommendations(self, title, top_n=10):
        """
        Get movie recommendations based on cosine similarity of features.
        
        Parameters:
        title (str): The title of the movie to find similar movies for.
        top_n (int): The number of top similar movies to return.
        
        Returns:
        pd.Series: The top N most similar movies.
        """
        # Find the index of the movie in the dataset
        idx = self.data[self.data['name'] == title].index[0]
        
        # Get the pairwise similarity scores for all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the top_n most similar movies
        sim_scores = sim_scores[1:top_n+1]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top N most similar movies
        return self.data['name'].iloc[movie_indices]
    
    def save_model(self, file_name):
        """
        Save the trained model to a file for later use.
        
        Parameters:
        file_name (str): The name of the file to save the model to.
        """
        import pickle
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, file_name):
        """
        Load a previously saved recommendation model from a file.
        
        Parameters:
        file_name (str): The name of the file to load the model from.
        
        Returns:
        MovieRecommendationModel: The loaded model.
        """
        import pickle
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
        return model

# Example Usage

# Step 1: Load the dataset
data = pd.read_csv('Comedy.csv')

# Step 2: Initialize and train the model
model = MovieRecommendationModel(data)

# # Step 3: Get recommendations for a specific movie
# movie_title = 'Guardians of the Galaxy'  # Example movie
# recommended_movies = model.get_recommendations(movie_title, top_n=10)

movie = input("Enter the movie name: ")
recommended_movies = model.get_recommendations(movie, top_n=10)
# Step 4: Display the recommendations
print(f"Movies similar to '{movie}':")
print(recommended_movies)

# Optional: Save the trained model to a file
model.save_model('movie_recommendation_model.pkl')

# Optional: Load the model from the file
loaded_model = MovieRecommendationModel.load_model('movie_recommendation_model.pkl')
