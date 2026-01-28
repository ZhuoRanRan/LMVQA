from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus
from VideoQA_Pipeline.align_multimodal_data import align_multimodal_data

video_name = "Lecture1"

user_question = "What are the learning objectives mentioned at the beginning of the lecture?"

multimodal_context = {
    "video_name": video_name,
    "video_type": "low-action",
    "video_duration": 380 
}

retriever = RagRetrieverMilvus(video_name)
if retriever.collection.num_entities == 0:
    print("⚠️ Vector DB is empty. Please run test_build_and_embed.py first.")
else:
    print(f"✅ Milvus collection contains {retriever.collection.num_entities} entries.")

top_chunks = retriever.query(user_question, top_k=5)

print("\n====== Top-5 Retrieved Chunks ======\n")
for i, chunk in enumerate(top_chunks):
    print(f"{i+1}. [{chunk['source']}] {chunk['timestamp']}: {chunk['text'][:150]}...\n")

prompt = align_multimodal_data(
    multimodal_context=multimodal_context,
    user_question=user_question,
    use_full_context=False,
    top_k=5
)

print("\n====== Final Constructed Prompt ======\n")
print(prompt)
