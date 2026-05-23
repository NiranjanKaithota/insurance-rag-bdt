import os
import sys

# Force PySpark to use the current virtual environment's Python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(binary_content):
    """Takes binary PDF data and returns extracted text."""
    try:
        import fitz
        pdf_document = fitz.open(stream=binary_content, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def chunk_text(text):
    """Splits the massive text into manageable, overlapping chunks."""
    if text.startswith("Error"): return []
    
    # We use overlapping chunks so clauses that cross paragraphs aren't lost
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def generate_embeddings(iterator):
    """Processes chunks and generates vector embeddings inside the Spark Workers."""
    from sentence_transformers import SentenceTransformer
    
    print("Loading Embedding Model on Worker...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for file_path, chunks in iterator:
        if not chunks:
            continue
        
        embeddings = model.encode(chunks).tolist()
        
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            # Create a clean, unique ID without weird file path characters
            clean_name = os.path.basename(file_path).replace(".pdf", "").replace("-", "_")
            chunk_id = f"{clean_name}_chunk_{i}"
            
            yield {
                "chunk_id": chunk_id, 
                "text": chunk, 
                "embedding": emb  # Now yielding the FULL 384-dimensional vector
            }

def push_to_milvus(partition_iterator):
    """Takes the vectorized data from Spark and pushes it to Milvus."""
    from pymilvus import connections, Collection
    
    # 1. Connect to Milvus from the Spark Worker
    connections.connect("default", host="127.0.0.1", port="19530")
    collection = Collection("insurance_policies")
    
    # 2. Batch the inserts for high performance
    batch = []
    for row in partition_iterator:
        batch.append(row)
        
        # Push to database in chunks of 100
        if len(batch) >= 100:
            collection.insert(batch)
            batch = []
            
    # Push any remaining chunks
    if batch:
        collection.insert(batch)

def main():
    from pymilvus import MilvusClient, DataType
    
    print("1. Initializing Milvus Database Schema...")
    client = MilvusClient(uri="tcp://127.0.0.1:19530")
    collection_name = "insurance_policies"
    
    # Drop the old collection if we are re-running the script
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        
    # Define the Table Blueprint the modern way
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=500)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=384)
    
    # Create an Index to make vector math blazing fast
    print("2. Building Vector Index...")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        params={"nlist": 128}
    )
    
    # Create Collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    print("3. Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("InsurancePolicyProcessor") \
        .master("local[2]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .config("spark.network.timeout", "3600s") \
        .config("spark.executor.heartbeatInterval", "600s") \
        .getOrCreate()

    hdfs_path = "hdfs://localhost:9000/data/raw/insurance_pdfs/*.pdf"
    
    # Execute the Pipeline
    print("4. Executing Big Data Pipeline (Read -> Chunk -> Embed -> Database)...")
    pdf_rdd = spark.sparkContext.binaryFiles(hdfs_path)
    text_rdd = pdf_rdd.mapValues(extract_text_from_pdf)
    chunked_rdd = text_rdd.mapValues(chunk_text)
    embedded_rdd = chunked_rdd.mapPartitions(generate_embeddings)
    
    # ACTION: This triggers the whole pipeline and pushes directly to Milvus!
    embedded_rdd.foreachPartition(push_to_milvus)
    
    # Load collection into memory for searching
    client.load_collection(collection_name=collection_name)
    
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETE!")
    print("Data successfully vectorized and stored in Milvus Vector DB.")
    print("="*50 + "\n")

    spark.stop()

if __name__ == "__main__":
    main()