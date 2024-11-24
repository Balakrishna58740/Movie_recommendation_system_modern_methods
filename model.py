import asyncio
import datetime
from typing import List
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from db_config import mongodb
import nltk

# Download the NLTK punkt tokenizer if not already installed
nltk.download("punkt")


class MovieRecommender:
    model_path = "movie_word2vec.model"

    @staticmethod
    async def fetch_data_from_db():
        try:
            movies = await mongodb.collections["Movies"].find().to_list(length=None)
            for movie in movies:
                movie["_id"] = str(movie["_id"])  # Convert ObjectId to string
            return movies
        except Exception as e:
            raise Exception(f"Error fetching data from database: {str(e)}")

    @staticmethod
    def preprocess_text(text: str) -> list:
        return word_tokenize(text.lower()) if isinstance(text, str) else []


    @staticmethod
    def get_similar_movies(movie_title: str, model: Word2Vec, data: pd.DataFrame, top_n: int = 5) -> List[str]:
        try:
            # Ensure 'tokens' column is generated
            if 'tokens' not in data.columns:
                data['description'] = data['genres'].apply(lambda x: " ".join(x)) + " " + data['movie_rated'].apply(lambda x: " ".join(x))
                data['tokens'] = data['description'].apply(MovieRecommender.preprocess_text)
            # Get the target movie vector
            target_vector = MovieRecommender.get_movie_vector(movie_title, model, data)
            similarities = []
            for _, row in data.iterrows():
                other_title = row["name"]
                if other_title == movie_title:
                    continue

                tokens = row["tokens"]
                if not isinstance(tokens, list):
                    continue

                try:
                    other_vector = MovieRecommender.get_movie_vector(other_title, model, data)
                    similarity = cosine_similarity([target_vector], [other_vector])[0][0]
                    similarities.append((other_title, similarity))
                except ValueError:
                    continue

            # Sort by similarity and return top N movies
            sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            return [movie[0] for movie in sorted_similarities[:top_n]]
        except Exception as e:
            raise Exception(f"Error getting similar movies: {str(e)}")
        
    @staticmethod
    def load_model() -> Word2Vec:
        try:
            model = Word2Vec.load(MovieRecommender.model_path)
            return model
        except Exception as e:
            raise Exception(f"Error loading the model: {str(e)}")
    @staticmethod
    def get_movie_vector(movie_title: str, model: Word2Vec, data: pd.DataFrame) -> list:
        try:
            # Find the row corresponding to the movie title
            movie_row = data[data['name'] == movie_title]
            if movie_row.empty:
                raise ValueError(f"Movie title '{movie_title}' not found in the dataset.")

            # Get the tokenized description
            tokens = movie_row['tokens'].values[0]

            # Ensure 'tokens' is a list and contains valid words
            if not isinstance(tokens, list):
                raise ValueError(f"Tokens for movie '{movie_title}' are not in the correct format (list).")

            # Calculate the average vector for the tokens
            word_vectors = []
            for word in tokens:
                if word in model.wv:
                    word_vectors.append(model.wv[word])
                else:
                    print(f"Warning: Word '{word}' not found in the model vocabulary.")  # Log the missing word

            if not word_vectors:
                raise ValueError(f"No valid word vectors found for movie '{movie_title}'.")

            # Average the word vectors
            movie_vector = sum(word_vectors) / len(word_vectors)
            return movie_vector
        except Exception as e:
            print(f"Error getting movie vector for '{movie_title}': {str(e)}")
            raise Exception(f"Error getting movie vector: {str(e)}")


    @staticmethod
    async def train_and_save_model():
        try:
            # Fetch data where `trained` field is false or missing
            movie_data = await mongodb.collections['Movies'].find({"Train": {"$ne": True}}).to_list(None)
            if not movie_data:
                return "No new data to train."

            data = pd.DataFrame(movie_data)

            # Train the model
            MovieRecommender.train_model(data)

            # After training, update the `trained` field for processed documents
            movie_ids = [item['_id'] for item in movie_data]
            await mongodb.collections['Movies'].update_many(
                {"_id": {"$in": movie_ids}},
                {"$set": {"Train": True}}
            )
            log_entry = {
                "timestamp": datetime.datetime.utcnow(),
                "records_trained": len(movie_ids),
                "status": "success",
            }
            await mongodb.collections["TrainLogs"].insert_one(log_entry)
            return "Model trained and saved successfully, and data marked as trained."
        except Exception as e:
            raise Exception(f"Error training and saving model: {str(e)}")

    @staticmethod
    def train_model(data: pd.DataFrame):
        try:
            # Ensure columns are strings and handle NaN values
            data["genres"] = data["genres"].fillna("").astype(str)
            data["movie_rated"] = data["movie_rated"].fillna("").astype(str)
            data["description"] = data["genres"] + " " + data["movie_rated"]
            data["tokens"] = data["description"].apply(MovieRecommender.preprocess_text)

            # Train the Word2Vec model
            model = Word2Vec(sentences=data["tokens"], vector_size=100, window=5, min_count=1, workers=4)
            model.save(MovieRecommender.model_path)
            return model
        except Exception as e:
            raise Exception(f"Error training the model: {str(e)}")