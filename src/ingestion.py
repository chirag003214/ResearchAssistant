import os
from llama_index.core import SimpleDirectoryReader, Document
from typing import List

def load_documents(data_dir: str = "data/raw") -> List[Document]:
    """
    Loads all documents from the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing PDFs.
        
    Returns:
        List[Document]: A list of LlamaIndex Document objects.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' not found. Please create it and add PDFs.")

    print(f"🔄 Loading documents from {data_dir}...")
    
    # SimpleDirectoryReader is the standard LlamaIndex loader.
    # It automatically handles PDFs, text files, and more.
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,  # Search subdirectories
        filename_as_id=True # Use filename as the unique ID for the doc
    )
    
    documents = reader.load_data()
    
    print(f"✅ Successfully loaded {len(documents)} document pages.")
    
    # Optional: specialized metadata cleanup can happen here
    # (e.g., cleaning up weird characters from filenames)
    
    return documents

if __name__ == "__main__":
    # This block allows you to test the ingestion script directly
    try:
        docs = load_documents()
        if docs:
            print(f"--- Sample Content from first page ---")
            print(f"Metadata: {docs[0].metadata}")
            print(f"Text Snippet: {docs[0].text[:500]}...")
    except Exception as e:
        print(f"Error: {e}")