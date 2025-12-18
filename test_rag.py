from src.ingestion import load_documents
from src.retriever import get_index

# 1. Load Docs
docs = load_documents()

# 2. Build Index (This will take a moment as it downloads the model)
index = get_index(docs)

# 3. Ask a question
engine = index.as_query_engine()
response = engine.query("What is the main topic of these documents?")
print("\n🤖 AI Answer:\n", response)