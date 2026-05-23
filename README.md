# 🏛️ Multi-Agent GraphRAG System for Indian Government Insurance Inference

**Project Context:** RV College of Engineering (RVCE) - Big Data Technology

## 📖 Overview
This project implements an enterprise-grade Big Data integration pipeline designed to ingest, process, and accurately query massive, unstructured government health insurance policies (e.g., PM-JAY, IRDAI Circulars). 

It utilizes a Retrieval-Augmented Generation (RAG) architecture to prevent LLM hallucinations, ensuring that all inferred eligibility and benefit answers are strictly grounded in official documents.

## 🏗️ Current Architecture (Phase 1 & 2 Completed)
The system currently operates a fully functional linear RAG pipeline using a local Dockerized cluster:

1. **Automated Ingestion (Apache NiFi):** Monitors local directories and securely transfers raw UPSC policy compilations and PDF circulars into the cluster.
2. **Distributed Storage (Hadoop HDFS):** Provides fault-tolerant persistence for the raw documents (`/data/raw/insurance_pdfs/`).
3. **Parallel Compute (Apache Spark):** Distributes the text-extraction and PyTorch-based vectorization (using `all-MiniLM-L6-v2`). Throttled to `local[2]` for stable local machine processing.
4. **Vector Database (Milvus):** Stores the 384-dimensional mathematical embeddings for millisecond-speed semantic similarity searches.
5. **Generation (Cohere API):** Utilizes the `command-r-08-2024` model, specifically optimized for long-context RAG, to synthesize factual answers.

## 🚀 Development Roadmap

### ✅ Sprint 1: Knowledge Base & Linear RAG (Completed)
- [x] Docker-compose cluster orchestration.
- [x] NiFi to HDFS automated routing.
- [x] Distributed chunking and embedding generation via Spark.
- [x] Milvus schema creation and semantic retrieval pipeline.
- [x] Secure API integration (`python-dotenv`) with Cohere.

### ⏳ Sprint 2: Hybrid Knowledge Graph (Next Up)
- [ ] Implement Entity & Relation extraction (Triplets).
- [ ] Connect Apache Spark to **Neo4j** Graph Database.
- [ ] Map complex policy rules (e.g., *Scheme A -> COVERS -> Disease B*).
- [ ] Upgrade retrieval query to pull from both Milvus (Vector) and Neo4j (Graph).

### ⏳ Sprint 3: Multi-Agent Orchestration
- [ ] Replace linear script with a stateful **LangGraph** swarm.
- [ ] Build **Analyzer Agent:** Determines user intent.
- [ ] Build **Validator Agent:** Cross-references retrieved data with live web searches to ensure policies haven't expired.
- [ ] Build **Synthesizer Agent:** Debates and drafts the final response.

### ⏳ Sprint 4: The Eligibility Matrix
- [ ] Implement structured output formatting.
- [ ] Ensure the final agent output is a highly structured, tabular "Eligibility Matrix" rather than conversational text.

## 🛠️ Tech Stack
* **Storage & Ingestion:** Apache Hadoop (HDFS), Apache NiFi
* **Processing:** Apache Spark, PySpark, PyTorch (`sentence-transformers`)
* **Databases:** Milvus (Vector), Neo4j (Graph - *Pending*)
* **AI & Orchestration:** Cohere Command-R, LangGraph (*Pending*)