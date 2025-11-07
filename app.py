import os
import json
import pandas as pd
import chromadb
from openai import OpenAI
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import re
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  

print("🚀 Loading environment variables...")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_store")
PRODUCTS_FILE = "product.xlsx"

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file")


client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = chroma_client.get_collection("products")
    print("✅ Loaded existing 'products' collection")
except Exception:
    print("⚠️ Collection not found, creating new one...")
    collection = chroma_client.create_collection("products")


if not os.path.exists(PRODUCTS_FILE):
    raise FileNotFoundError(f"❌ Excel file not found at {PRODUCTS_FILE}")

df = pd.read_excel(PRODUCTS_FILE).fillna("")
df["normalized_title"] = df["Title"].astype(str).str.lower().str.strip()
df["normalized_sku"] = df["Variant SKU"].astype(str).str.lower().str.strip()
df["normalized_handle"] = df["Handle"].astype(str).str.lower().str.strip()

print(f"✅ Loaded {len(df)} products from Excel")


def find_exact_match(query: str):
    q = query.lower().strip()
    mask = (
        df["normalized_title"].str.contains(re.escape(q), na=False)
        | df["normalized_sku"].str.contains(re.escape(q), na=False)
        | df["normalized_handle"].str.contains(re.escape(q), na=False)
    )
    results = df[mask]

    if not results.empty:
        matches = []
        for _, row in results.iterrows():
            matches.append({
                "title": row["Title"],
                "sku": row["Variant SKU"],
                "handle": row["Handle"],
                "type": row.get("Type", ""),
                "tags": row.get("Tags", ""),
                "price": row.get("Variant Price", ""),
                "body_html": row.get("Body HTML", "")
            })
        return matches
    return None


def is_out_of_context(query: str):
    q = query.lower().strip()

    unrelated_keywords = [
        "apple", "banana", "fruit", "vegetable", "food", "drink",
        "clothes", "shirt", "pants", "shoes", "bag", "tv", "phone",
        "laptop", "computer", "tablet", "software", "music", "movie",
        "perfume", "watch", "cosmetic", "medicine", "furniture", "toy"
    ]
    if any(word in q for word in unrelated_keywords):
        print("🛑 Unrelated keyword detected.")
        return True

    related_terms = [
        "tractor", "branson", "filter", "plug", "hose", "pump", "valve",
        "engine", "part", "bolt", "screw", "bearing", "gasket", "seat",
        "axle", "oil", "fuel", "hydraulic", "transmission", "steering",
        "clutch", "belt", "sensor", "lever", "arm"
    ]
    if any(word in q for word in related_terms):
        print("✅ Query looks related to our domain.")
        return False

    try:
        print("🤖 Checking relevance using GPT...")
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=f"Is the following query about tractor parts, Branson tractors, or mechanical components? Answer 'Yes' or 'No'. Query: {query}"
        )
        ans = resp.output[0].content[0].text.lower()
        return "no" in ans
    except Exception as e:
        print("⚠️ GPT check failed:", e)
        return False


def semantic_search(query: str):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    query_embedding = response.data[0].embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    formatted = []

    if results.get("ids"):
        for i in range(len(results["ids"][0])):
            item = results["metadatas"][0][i]
            formatted.append({
                "title": item.get("title", ""),
                "type": item.get("type", ""),
                "sku": item.get("sku", ""),
                "price": item.get("price", ""),
                "tags": item.get("tags", ""),
                "body_html": item.get("body_html", ""),
                "handle": item.get("handle", ""),
                "score": results["distances"][0][i]
            })
    return formatted


@app.route("/semantic-search", methods=["POST"])
def hybrid_search():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "No query provided"}), 400

        print(f"\n🔍 Incoming query: {query}")

        # 1️⃣ Out-of-context detection
        if is_out_of_context(query):
            return jsonify({
                "query": query,
                "message": "We do not have such products.",
                "results": [],
                "source": "none"
            })

        # 2️⃣ Exact match
        exact_results = find_exact_match(query)
        if exact_results:
            print("🎯 Exact match found!")
            return jsonify({"query": query, "results": exact_results, "source": "exact"})

        # 3️⃣ Semantic fallback
        print("🧠 No exact match — running semantic search...")
        semantic_results = semantic_search(query)
        if semantic_results:
            return jsonify({"query": query, "results": semantic_results, "source": "semantic"})

        # 4️⃣ Nothing found
        print("⚙️ No match at all.")
        return jsonify({
            "query": query,
            "message": "No relevant products found.",
            "results": [],
            "source": "none"
        })

    except Exception as e:
        print("❌ Error during search:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Running Flask app on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
