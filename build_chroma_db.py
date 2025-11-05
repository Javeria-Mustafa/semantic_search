import os
import time
import pandas as pd
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from random import randint

# Load environment variables
load_dotenv()

# ✅ Initialize OpenAI and Chroma clients
client = OpenAI(api_key="" + os.getenv("OPENAI_API_KEY", ""))
chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "chroma_store1"))
collection = chroma_client.get_or_create_collection("products")

# ✅ Load Excel
df = pd.read_excel("Sample Data Full Export (2).xlsx").fillna("")

def make_text(row):
    return (
        f"{row['Title']} {row['Body HTML']} "
        f"Type: {row['Type']} "
        f"Tags: {row['Tags']} "
        f"Price: {row['Variant Price']} "
        f"SKU: {row['Variant SKU']} "
        f"Handle: {row['Handle']}"
    )

texts = [make_text(row) for _, row in df.iterrows()]

metadatas = [
    {
        "title": row["Title"],
        "type": str(row["Type"]),
        "sku": str(row["Variant SKU"]),
        "price": str(row["Variant Price"]),
        "tags": str(row["Tags"]),
        "body_html": row["Body HTML"],
        "handle": str(row["Handle"]),   
    }
    for _, row in df.iterrows()
]

ids = [f"item-{i}" for i in range(len(df))]

# ✅ Auto-batch size tuning
def get_batch_size(total, max_allowed):
    """Pick a safe batch size automatically."""
    return min(max(10, total // 20), max_allowed)

OPENAI_BATCH = get_batch_size(len(texts), 100)   # usually 50–100
CHROMA_BATCH = get_batch_size(len(texts), 5000)  # <= 5461 limit

print(f"🧠 Using OpenAI batch size: {OPENAI_BATCH}, Chroma batch size: {CHROMA_BATCH}")

# ✅ Embed texts in batches
all_embeddings = []

for i in tqdm(range(0, len(texts), OPENAI_BATCH), desc="Generating embeddings"):
    batch_texts = texts[i:i + OPENAI_BATCH]

    for retry in range(3):  # retry logic
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=batch_texts
            )
            break
        except Exception as e:
            wait = randint(2, 5)
            print(f"⚠️ Retry {retry+1}/3 after {wait}s due to error: {e}")
            time.sleep(wait)
    else:
        raise RuntimeError("❌ Embedding failed repeatedly. Stopping.")

    batch_embeddings = [d.embedding for d in response.data]
    all_embeddings.extend(batch_embeddings)

# ✅ Add to Chroma in batches
for i in tqdm(range(0, len(all_embeddings), CHROMA_BATCH), desc="Adding to Chroma"):
    collection.add(
        embeddings=all_embeddings[i:i + CHROMA_BATCH],
        documents=texts[i:i + CHROMA_BATCH],
        metadatas=metadatas[i:i + CHROMA_BATCH],
        ids=ids[i:i + CHROMA_BATCH]
    )

print("\n✅ All embeddings stored in Chroma successfully!")
