import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, InsertOne
from pymongo.errors import ConnectionFailure, BulkWriteError
import time

from product_embedding_generator import generate_product_embeddings_in_memory

load_dotenv()

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
DATABASE_NAME = os.getenv("DATABASE_NAME", "your_vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products")

EMBEDDING_COLUMN_NAME = os.getenv("EMBEDDING_COLUMN_NAME", "embedding")
PRODUCT_ID_COLUMN_NAME = os.getenv("PRODUCT_ID_COLUMN_NAME", "product_id")

def connect_mongodb(mongo_uri):
    if not mongo_uri:
        print("Error: MongoDB Connection String not provided in environment variables.")
        return None
    try:
        print("Attempting to connect to MongoDB Atlas...")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("MongoDB Atlas connection successful!")
        return client
    except ConnectionFailure as e:
        print(f"MongoDB Atlas connection failed: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during MongoDB connection: {e}")
        return None

def prepare_documents_from_dataframe(data_df: pd.DataFrame, product_id_col: str, embedding_col: str):
    if data_df is None or data_df.empty:
        print("No data DataFrame provided or DataFrame is empty for document preparation.")
        return None

    print(f"Preparing {len(data_df)} documents for insertion from DataFrame...")

    required_cols = [product_id_col, "product_name", embedding_col]
    if not all(col in data_df.columns for col in required_cols):
         missing = [col for col in required_cols if col not in data_df.columns]
         print(f"Error: Missing required columns in DataFrame for document preparation: {missing}")
         return None

    if data_df[product_id_col].dtype != np.int64:
        print(f"Converting '{product_id_col}' column from '{data_df[product_id_col].dtype}' to int64...")
        initial_rows = len(data_df)
        data_df[product_id_col] = pd.to_numeric(data_df[product_id_col], errors='coerce')
        data_df.dropna(subset=[product_id_col], inplace=True)
        if len(data_df) < initial_rows:
            print(f"Warning: Dropped {initial_rows - len(data_df)} rows with invalid (non-numeric) '{product_id_col}' values during preparation.")
        data_df[product_id_col] = data_df[product_id_col].astype(np.int64)
        print(f"Conversion successful. '{product_id_col}' dtype is now {data_df[product_id_col].dtype}")

    documents_to_insert = []
    for index, row in data_df.iterrows():
        document = {
            "_id": int(row[product_id_col]),
            "product_name": row["product_name"],
            embedding_col: row[embedding_col].tolist() if isinstance(row[embedding_col], np.ndarray) else row[embedding_col]
        }
        documents_to_insert.append(document)

    if not documents_to_insert:
         print("No valid documents prepared for insertion.")
         return None

    print(f"Prepared {len(documents_to_insert)} documents for insertion.")
    return documents_to_insert


def ingest_data_into_mongodb(mongo_client, db_name, collection_name, documents):
    if mongo_client is None:
        print("Cannot ingest data: MongoDB client is None.")
        return False
    if not documents:
        print("No documents to ingest.")
        return False

    try:
        db = mongo_client[db_name]
        collection = db[collection_name]

        print(f"Starting batch insertion into collection '{collection_name}' in database '{db_name}'...")

        batch_size = 1000
        num_documents = len(documents)

        inserted_count = 0
        start_time = time.time()

        for i in range(0, num_documents, batch_size):
            batch_documents = documents[i:i + batch_size]
            try:
                insert_result = collection.insert_many(batch_documents, ordered=False)
                batch_inserted_count = len(insert_result.inserted_ids)
                inserted_count += batch_inserted_count
                print(f"Inserted batch {i//batch_size + 1}/{(num_documents + batch_size - 1)//batch_size} ({batch_inserted_count}/{len(batch_documents)} documents inserted).")

            except BulkWriteError as bwe:
                inserted_count += bwe.details['nInserted']
                print(f"BulkWriteError in batch starting at index {i}: {bwe.details['nInserted']} documents inserted, {len(bwe.details.get('writeErrors', []))} errors.")
            except Exception as batch_e:
                print(f"An unexpected error occurred during batch insertion starting at index {i}: {batch_e}")

        end_time = time.time()
        print(f"\nBatch insertion complete. Total documents processed: {num_documents}. Total documents inserted: {inserted_count}.")
        print(f"Ingestion took {end_time - start_time:.2f} seconds.")

        return True

    except Exception as e:
        print(f"An error occurred during the ingestion process: {e}")
        return False


if __name__ == "__main__":
    print("--- MongoDB Atlas Data Ingestion Script (Direct from Embeddings) ---")

    target_merchant_id = '22b725f2-f1bb-411f-902d-554905352af4'
    raw_json_source_path = './sns.products.json'

    mongo_client = connect_mongodb(MONGODB_CONNECTION_STRING)

    if mongo_client:
        try:
            print(f"\nCalling embedding generator for merchant_id: {target_merchant_id}...")
            embeddings_df = generate_product_embeddings_in_memory(
                merchant_id=target_merchant_id,
                raw_json_input_path=raw_json_source_path
            )

            if embeddings_df is not None and not embeddings_df.empty:
                print(f"Successfully received embeddings DataFrame with {len(embeddings_df)} rows.")

                documents_to_insert = prepare_documents_from_dataframe(
                    embeddings_df, PRODUCT_ID_COLUMN_NAME, EMBEDDING_COLUMN_NAME
                )

                if documents_to_insert:
                    ingest_success = ingest_data_into_mongodb(
                        mongo_client, DATABASE_NAME, COLLECTION_NAME, documents_to_insert
                    )

                    if ingest_success:
                        print("\nData ingestion process completed.")
                    else:
                        print("\nData ingestion process failed.")
                else:
                    print("\nFailed to prepare valid documents for insertion from embeddings DataFrame.")
            else:
                print("\nNo embeddings DataFrame received or it was empty. Skipping ingestion.")

        finally:
            if mongo_client:
                mongo_client.close()
                print("MongoDB connection closed.")

    else:
        print("Exiting script due to MongoDB connection failure.")

    print("--- End of Script ---")