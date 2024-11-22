from typing import List
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pandas as pd
from model import MovieRecommender
from db_config import mongodb

app = FastAPI()
scheduler = AsyncIOScheduler()

# MongoDB connection management
@app.on_event("startup")
async def startup():
    try:
        # Connect to MongoDB
        await mongodb.connect(
            uri="mongodb+srv://pukar:pukarpass@cluster0.2xqtwbl.mongodb.net/DataScience",
            db_name="DataScience",
        )
        print("MongoDB connected successfully.")

        # Add the job to the scheduler
        scheduler.add_job(
            MovieRecommender.train_and_save_model,
            "cron",
            hour="*",
            minute="*",
        )
        scheduler.start()
        print("Scheduler started")
    except Exception as e:
        print(f"Error during startup: {str(e)}")


@app.on_event("shutdown")
async def shutdown():
    try:
        # Shutdown MongoDB connection
        await mongodb.close()
        print("MongoDB connection closed.")

        # Shutdown the scheduler
        scheduler.shutdown()
        print("Scheduler shutdown.")
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Recommendation API!"}


@app.post("/data")
async def create_data(data: List[dict]):
    try:
        if "Movies" not in mongodb.collections:
            return {"error": "Movies collection not found"}
        response = await mongodb.collections["Movies"].insert_many(data)
        return {"Message": "Data created successfully", "inserted_ids": [str(id_) for id_ in response.inserted_ids]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data")
async def fetch_data():
    try:
        return await MovieRecommender.fetch_data_from_db()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendation/{movie_title}")
async def get_recommendations(movie_title: str) -> List[str]:
    try:
        # Fetch data from the database
        data = await MovieRecommender.fetch_data_from_db()
        # Convert to DataFrame
        data_df = pd.DataFrame(data)
        # Load the pre-trained model
        model = MovieRecommender.load_model()
        # Get recommendations
        recommendations = MovieRecommender.get_similar_movies(movie_title, model, data_df)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")