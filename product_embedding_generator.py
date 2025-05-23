import json
import os
import time
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def generate_product_embeddings_in_memory(
    merchant_id: str,
    raw_json_input_path='./sns.products.json'
) -> pd.DataFrame:
    print(f"--- Starting In-Memory Product Data Processing for Merchant: {merchant_id} ---")
    start_time = time.time()

    print(f"\nStep 0: Filtering JSON data by merchant_id '{merchant_id}' (in memory)...")
    try:
        with open(raw_json_input_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)

        input_data = raw_data if isinstance(raw_data, list) else [raw_data]
        filtered_items = [item for item in input_data if item.get('merchant_id') == merchant_id]

        if not filtered_items:
            print(f"No items found for merchant_id '{merchant_id}' after filtering. Exiting process.")
            return None

        print(f"Found {len(filtered_items)} record(s) with merchant_id: {merchant_id}")

    except FileNotFoundError:
        print(f"Error: The raw input file '{raw_json_input_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{raw_json_input_path}' contains invalid JSON.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON filtering: {str(e)}")
        return None

    print(f"\nStep 1: Creating DataFrame and preprocessing data for merchant '{merchant_id}'...")
    try:
        data = pd.DataFrame(filtered_items)

        required_cols = ['product_id', 'handle', 'body_html']
        for col in required_cols:
            if col not in data.columns:
                data[col] = ''

        data.rename(columns={'product_id': 'Product_ID',
                             'handle': 'Handle',
                             'body_html': 'Body_HTML'}, inplace=True)

        data['Description'] = data['Body_HTML'].astype(str).str.replace(r'^<p>', '', regex=True).str.replace(r'</p>$', '', regex=True)
        data['Handle'] = data['Handle'].astype(str).str.replace('-', ' ', regex=False)
        data['Handle'] = data['Handle'].astype(str).str.replace('100', '100%', regex=False)

        data.rename(columns={'Handle': 'productDisplayName'}, inplace=True)

        data['text'] = data['productDisplayName'].fillna('') + ' ' + data['Description'].fillna('')
        print("Data preprocessed successfully and text combined for embeddings.")

    except Exception as e:
        print(f"An unexpected error occurred during DataFrame creation or preprocessing: {str(e)}")
        return None

    print("\nStep 2: Loading SentenceTransformer model and generating embeddings...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')

        product_text = data['text'].tolist()

        embeddings = model.encode(product_text, show_progress_bar=True, batch_size=32)
        print(f"Generated {len(embeddings)} embeddings.")

        results = []
        for i, row in enumerate(data.itertuples(index=False)):
            product_id = str(row.Product_ID)
            product_name = str(row.productDisplayName)
            prod_embedding = embeddings[i]

            results.append({
                'product_id': product_id,
                'product_name': product_name,
                'embedding': prod_embedding
            })

        embedding_df = pd.DataFrame(results)

    except Exception as e:
        print(f"An unexpected error occurred during embedding generation: {str(e)}")
        return None

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n--- In-Memory Processing Completed for Merchant {merchant_id} in {elapsed_time:.2f} seconds ---")

    return embedding_df

if __name__ == "__main__":
    target_merchant_id = '22b725f2-f1bb-411f-902d-554905352af4'
    embeddings_df = generate_product_embeddings_in_memory(merchant_id=target_merchant_id)

    if embeddings_df is not None:
        print(f"\nSuccessfully generated embeddings DataFrame with {len(embeddings_df)} rows.")
    else:
        print("\nFailed to generate embeddings DataFrame.")