import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.embeddings import OpenAIEmbeddings , HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document
import time

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.2
        )



template = """
You are a retail recommendation expert specializing in identifying **complementary products** for items sold on an Ecommerce platform.

CONTEXT:
- Original Product: {original_product_name}
- Available Categories :
{category}

Your task is to suggest **up to 5 complementary products** that are typically purchased, worn, or used **together** with the original product. Your recommendations must be selected **only from the provided categories**. Rank them in **decreasing order of complementary relevance**.

Focus your suggestions on:
- Functional utility (e.g., matching bottomwear, required underlayers)
- Styling and enhancement (e.g., accessories, color coordination)
- Target usage (festive, daily wear, casual, formal, age relevance)
- The original product’s intended gender, style, and cultural context

📌 INSTRUCTIONS:
- Only recommend items from the listed categories within `{category}`
- Do **NOT** include substitutes or near-identical products
- Do **NOT** include items from categories not listed
- Prioritize items that help **complete, accessorize, or enhance** the product
- **For recommended products, use specific item names or types (e.g., "Slim Fit Chinos", "Statement Necklace") rather than just the general category name (e.g., not just "Bottoms - Men", "Necklaces").**

📊 SCORING:
- Assign a **complementary score between 0.80 and 1.00**
- Only include items with a score **≥ 0.85**, unless slightly lower but very relevant
- Fewer than 5 is okay — **precision matters more than quantity**
- List in **descending order of complementary score**

📎 OUTPUT FORMAT:
For each complementary product:
- **Product Name**
- **Brief Description** (1-2 lines describing the item itself)
- **Complementary Score** (e.91)

Do not explain how it complements the original product.

🔍 EXAMPLES:

1.  Original: **Luxury Moisturizing Shampoo (500ml)**
    Category List:
    -   Personal Care: ["Conditioners", "Hair Masks", "Serums", "Body Wash"]
    -   Accessories: ["Hairbands", "Clips"]
    -   Appliances: ["Hair Dryers", "Straighteners"]

    Suggested Complementary Products:
    -   **Matching Deep Conditioner**
        -   **Brief Description**: A rich, hydrating conditioner formulated to work with the shampoo.
        -   **Complementary Score**: 0.96
    -   **Leave-In Hair Serum**
        -   **Brief Description**: Lightweight serum that smooths hair and tames frizz post-wash.
        -   **Complementary Score**: 0.89

2.  Original: **Men's Casual Cotton T-Shirt**
    Category List:
    -   Apparel: ["Jeans", "Shorts", "Jackets", "Sweatshirts"]
    -   Accessories: ["Sneakers", "Caps", "Socks", "Belts"]

    Suggested Complementary Products:
    -   **Comfort Fit Denim Jeans**
        -   **Brief Description**: Classic blue jeans with a relaxed fit for everyday comfort.
        -   **Complementary Score**: 0.96
    -   **White Casual Sneakers**
        -   **Brief Description**: Lightweight lace-up shoes that suit casual and semi-casual wear.
        -   **Complementary Score**: 0.91

{format_instructions}
"""

class ProductRelationship(BaseModel):
    product_name: str = Field(description="Name of the recommended product.")
    product_description: str = Field(description="Describe the product and explain what it is used for.")
    score: float = Field(ge=0.0, le=1.0, description="A score (0 to 1) indicating how complementary or similar the product is.")

class ProductRecommendations(BaseModel):
    complementary_products: List[ProductRelationship] = Field(
        description="List of complementary products"
    )

parser = PydanticOutputParser(pydantic_object=ProductRecommendations)

prompt = ChatPromptTemplate.from_template(
    template=template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

recommendation_chain = prompt | llm | parser

def get_complementary(
    product_name: str,
    category: Dict[str, List[str]],
    llm=None
) -> Dict[str, List[Dict[str, Any]]]:

    try:
        global recommendation_chain
    except NameError:
         print("Error: 'recommendation_chain' is not defined. Ensure LLM, prompt, and parser are set up.")
         return {"complementary_products": []}

    if llm is None:
         print("Error: LLM is not provided or initialized.")
         return {"complementary_products": []}

    print(f"Generating complementary products for: {product_name}")

    try:
        llm_response = recommendation_chain.invoke({
            "original_product_name": product_name,
            "category": category
        })

        complementary_list = llm_response.complementary_products

        sorted_complementary = sorted(
             complementary_list,
             key=lambda x: x.score,
             reverse=True
        )

        return {"complementary_products": [item.model_dump() for item in sorted_complementary]}

    except Exception as e:
        print(f"An error occurred during LLM invocation or parsing: {e}")
        return {"complementary_products": []}


# if __name__ == "__main__":
#     product_to_recommend_for = "Men's Casual Cotton T-Shirt"
#     available_categories = {
#         "Apparel": ["Jeans", "Shorts", "Jackets", "Sweatshirts"],
#         "Accessories": ["Sneakers", "Caps", "Socks", "Belts"]
#     }

#     llm_start_time = time.time()
#     recommendations = get_complementary(
#         product_name=product_to_recommend_for,
#         category=available_categories,
#         llm=llm
#     )
#     llm_end_time = time.time()
#     print(f"LLM response time: {llm_end_time - llm_start_time:.2f} seconds")

#     # if recommendations and recommendations["complementary_products"]:
#     #     print(f"\nComplementary products for '{product_to_recommend_for}':")
#     #     for item in recommendations["complementary_products"]:
#     #         print(f"- {item.get('product_name', 'N/A')} (Score: {item.get('score', 0.0):.2f})")
#     #         print(f"  Description: {item.get('product_description', 'N/A')}")
#     # else:
#     #     print(f"\nCould not generate recommendations for '{product_to_recommend_for}'.")