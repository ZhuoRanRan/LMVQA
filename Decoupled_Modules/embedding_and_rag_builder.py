import os
from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus

def embedding_and_rag_builder(video_name: str, overwrite: bool = False, top_k: int = 50):
    """
    Create or refresh vector embeddings for a given video's multimodal chunks,
    and optionally query to preview top_k similar chunks for debugging or testing.

    Args:
        video_name (str): The name of the video (e.g., 'Lecture1').
        overwrite (bool): Whether to force delete and recreate Milvus collection.
        top_k (int): Number of top similar chunks to preview (if needed).
    """
    retriever = RagRetrieverMilvus(video_name, overwrite=overwrite)

    print("🔧 Building vector embeddings...")
    retriever.build_from_chunks()

    print(f"🔍 Querying Milvus for a test question (top {top_k})")
    results = retriever.query("What is the main topic?", top_k=top_k)
    for i, chunk in enumerate(results):
        print(f"[{i+1}] ({chunk['source']}) {chunk['timestamp']}: {chunk['text']}")
