from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus

video_name = "Lecture1"
question = "Describe this video."

retriever = RagRetrieverMilvus(video_name)
top_chunks = retriever.query(question, top_k=20)

print("\n====== Top-K Chunks Retrieved ======\n")
for i, chunk in enumerate(top_chunks):
    print(f"{chunk['text']}...\n")
