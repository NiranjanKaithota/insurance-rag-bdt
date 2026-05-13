from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

def search_policies(question):
    print("\n" + "="*50)
    print(f"🧠 Processing Question: '{question}'")
    print("="*50)
    
    # 1. Load the exact same embedding model to ensure the math matches
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert the user's question into a 384-dimensional vector
    question_embedding = model.encode([question]).tolist()[0]

    # 2. Connect to Milvus
    print("Connecting to Milvus Database...")
    client = MilvusClient(uri="tcp://127.0.0.1:19530")
    collection_name = "insurance_policies"

    # 3. Perform the Vector Similarity Search
    print("Searching for the 3 most relevant policy chunks...")
    search_res = client.search(
        collection_name=collection_name,
        data=[question_embedding],
        limit=3,  # Bring back the top 3 closest matches
        output_fields=["text", "chunk_id"],  # We want the actual text back!
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
    )

    # 4. Display the Results
    print("\n" + "="*50)
    print("🎯 TOP MATCHES FOUND:")
    print("="*50)
    
    # search_res is a list of lists. We grab the first list of hits.
    for i, hit in enumerate(search_res[0]):
        print(f"\nMATCH {i+1} (Similarity Score: {hit['distance']:.4f})")
        print(f"Source: {hit['entity']['chunk_id']}")
        print(f"Text Preview: {hit['entity']['text'][:400]}...\n")
        print("-" * 50)

if __name__ == "__main__":
    # Test Question (Focusing on the medical/health schemes from your PDF)
    user_question = "What are the details of the Ayushman Bharat Health Infrastructure Mission?"
    search_policies(user_question)