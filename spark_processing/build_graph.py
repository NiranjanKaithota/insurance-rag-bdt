import os
import sys
from pyspark.sql import SparkSession
from neo4j import GraphDatabase
import spacy

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# ---------------------------------------------------------
# 1. TEXT EXTRACTION (Reusing our PDF logic)
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_content):
    import fitz # PyMuPDF
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ---------------------------------------------------------
# 2. NLP ENTITY EXTRACTION & NEO4J LOADING
# ---------------------------------------------------------
def process_and_push_to_neo4j(partition_iterator):
    """Worker function to run Cohere LLM and push semantic relationships to Neo4j."""
    import cohere
    import json
    import time
    import re
    from dotenv import load_dotenv
    
    # Load environment variables on the worker node
    load_dotenv()
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        print("⚠️ COHERE_API_KEY missing on worker. Skipping partition.")
        return
        
    co = cohere.Client(cohere_key)
    
    # Connect to Neo4j
    URI = "bolt://127.0.0.1:7687"
    AUTH = ("neo4j", "insurance_graph_password")
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            for file_path, chunks in partition_iterator:
                if not chunks:
                    continue
                
                source_doc = os.path.basename(file_path)
                
                # Create the Source Document Node
                session.run("MERGE (d:Document {name: $name})", name=source_doc)

                for chunk in chunks:
                    # 1. PROMPT THE LLM TO EXTRACT TRIPLETS
                    prompt = f"""
                    You are an expert data extractor. Read the following policy text and extract the key logical relationships as a strict JSON list of objects.
                    Each object must have exactly three keys: 'subject', 'relation', 'object'.
                    The 'relation' must be a single UPPERCASE verb with underscores (e.g., COVERS, ELIGIBLE_FOR, REQUIRES_DOCUMENT).
                    Do not include any text outside the JSON array.
                    
                    Text: {chunk[:1500]}
                    """
                    
                    try:
                        # Call Cohere
                        response = co.chat(
                            model='command-r-08-2024',
                            message=prompt,
                            temperature=0.1
                        )
                        
                        # Clean the response to extract just the JSON
                        raw_text = response.text
                        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
                        
                        if json_match:
                            triplets = json.loads(json_match.group(0))
                            
                            # 2. PUSH TRIPLETS TO NEO4J
                            for t in triplets:
                                sub = str(t.get('subject', '')).strip().upper()
                                rel = str(t.get('relation', '')).strip().upper().replace(" ", "_")
                                obj = str(t.get('object', '')).strip().upper()
                                
                                if not sub or not rel or not obj:
                                    continue
                                    
                                # Cypher query to dynamically create relationships
                                # Note: Neo4j requires relationship types to be injected directly into the string
                                cypher_query = f"""
                                MERGE (s:Entity {{name: $sub}})
                                MERGE (o:Entity {{name: $obj}})
                                MERGE (s)-[:{rel}]->(o)
                                MERGE (d:Document {{name: $doc_name}})
                                MERGE (d)-[:SOURCE_OF]->(s)
                                """
                                session.run(
                                    cypher_query, 
                                    sub=sub, 
                                    obj=obj, 
                                    doc_name=source_doc
                                )
                                
                        # Anti-Rate-Limit Sleep (Very important for free API tiers!)
                        time.sleep(2) 
                        
                    except Exception as e:
                        print(f"⚠️ Extraction failed for a chunk: {e}")
                        time.sleep(5) # Back off if we hit a rate limit error

def main():
    print("1. Initializing Spark Session for Graph Processing...")
    # Throttled to local[2] to protect your RAM during NLP processing!
    spark = SparkSession.builder \
        .appName("InsuranceGraphBuilder") \
        .master("local[2]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .getOrCreate()

    hdfs_path = "hdfs://localhost:9000/data/raw/insurance_pdfs/*.pdf"
    
    print("2. Reading PDFs from HDFS...")
    pdf_rdd = spark.sparkContext.binaryFiles(hdfs_path)
    
    print("3. Extracting and Chunking Text...")
    text_rdd = pdf_rdd.mapValues(extract_text_from_pdf)
    chunked_rdd = text_rdd.mapValues(chunk_text)
    
    print("4. Running NLP Extraction and Loading to Neo4j...")
    # Trigger the heavy NLP pipeline and push to the Graph DB
    chunked_rdd.foreachPartition(process_and_push_to_neo4j)

    print("\n" + "="*50)
    print("✅ GRAPH PIPELINE COMPLETE!")
    print("Entities and Relationships successfully mapped in Neo4j.")
    print("="*50 + "\n")

    spark.stop()

if __name__ == "__main__":
    main()