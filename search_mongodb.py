import os
from dotenv import load_dotenv
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from embedding_utils import get_embedding, VECTOR_DIMENSION
from typing import List, Dict, Any
from llm_service import get_complementary
from langchain_google_genai import ChatGoogleGenerativeAI
import time

load_dotenv()

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
DATABASE_NAME = os.getenv("DATABASE_NAME", "your_vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products")
VECTOR_SEARCH_INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX_NAME", "vector_index")
EMBEDDING_FIELD_NAME = os.getenv("EMBEDDING_FIELD_NAME", "embedding")

llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.2
        )



def find_similar_products_mongodb(query_text: str, top_k: int = 5) -> List[Dict[str, Any]] | None:
    if not MONGODB_CONNECTION_STRING:
        print("Error: MongoDB Connection String not provided.")
        return None

    query_embedding = get_embedding(query_text)

    if query_embedding is None:
        print("Failed to generate query embedding using embedding_utils.")
        return None

    if len(query_embedding) != VECTOR_DIMENSION:
         print(f"Error: Generated query embedding has incorrect dimension. Expected {VECTOR_DIMENSION}, got {len(query_embedding)}.")
         return None

    mongo_client = None
    search_results = []

    try:
        print("Attempting to connect to MongoDB Atlas for search...")
        client = MongoClient(MONGODB_CONNECTION_STRING, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("MongoDB Atlas connection successful!")

        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        print(f"Performing vector search on collection '{COLLECTION_NAME}' using index '{VECTOR_SEARCH_INDEX_NAME}'...")

        num_candidates = max(top_k * 10, 50)

        pipeline = [
            {
                '$vectorSearch': {
                    'index': VECTOR_SEARCH_INDEX_NAME,
                    'path': EMBEDDING_FIELD_NAME,
                    'queryVector': query_embedding,
                    'numCandidates': num_candidates,
                    'limit': top_k
                }
            },
            {
                '$project': {
                    "_id": 1,
                    "product_name": 1,
                    "score": { '$meta': 'vectorSearchScore' }
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        print(f"Vector search completed. Found {len(results)} similar products.")

        return results

    except ConnectionFailure as e:
        print(f"MongoDB Atlas connection failed during search: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during MongoDB vector search: {e}")
        return None
    finally:
        if mongo_client:
            client.close()
            print("MongoDB connection closed.")


if __name__ == "__main__":


    product_to_recommend_for = "Men's Casual Cotton T-Shirt"
    available_categories = {
        "Apparel": ["Jeans", "Shorts", "Jackets", "Sweatshirts"],
        "Accessories": ["Sneakers", "Caps", "Socks", "Belts"]
    }
    llm_start = time.time()
    recommendations = get_complementary(
        product_name=product_to_recommend_for,
        category=available_categories,
        llm=llm
    )
    llm_end = time.time()
    llm_time = llm_end - llm_start
    print(f"LLM Time: {llm_time:.2f} seconds")


    search_time_start = time.time()

    recommended_products = []
    if recommendations and recommendations["complementary_products"]:
        for item in recommendations["complementary_products"]:
            com_product_name = item.get('product_name', 'N/A')
            com_product_description = item.get('product_description', 'N/A')
            product_text = f"{com_product_name} {com_product_description}"

            # print("--- Starting MongoDB Atlas Vector Search Example ---")

            # print(f"Searching for similar products to: '{com_product_name}'")

            

            similar_items = find_similar_products_mongodb(product_text , top_k = 1)



            

            if similar_items:
                for item in similar_items:
                    item_id = item.get("_id", "N/A")
                    item_name = item.get("product_name", "N/A")
                    item_score = item.get("score", 0.0)
                    # print(f"  ID: {item_id}, Name: '{item_name}', Score: {item_score:.4f}")
                    recommended_products.append({
                        "product_id": item_id,
                        "product_name": item_name,
                        "score": item_score
                    })
            else:
                print("\nCould not find similar items or an error occurred during search.")

            # print("--- End of Search Example ---")

    print(f"Recommended Products: {recommended_products}")

    search_time_end = time.time()
    search_time = search_time_end - search_time_start
    print(f"Search Time: {search_time:.2f} seconds")