import os
from Decoupled_Modules.information_extraction_builder import information_extraction_builder
from Decoupled_Modules.embedding_and_rag_builder import embedding_and_rag_builder
from Decoupled_Modules.qa_answering_builder import qa_answering_builder

def test_pipeline_on_video(video_path: str, query: str, overwrite: bool = False):
    print(f"\n🎬 Testing pipeline on video: {video_path}\n")

    # Step 1: Run multimodal extraction
    print("🧠 Step 1: Extracting multimodal information...")
    chunk_path = information_extraction_builder(video_path, overwrite=overwrite)
    print(f"✅ Chunks saved to: {chunk_path}")

    # Step 2: Build embeddings into Milvus
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print("\n📊 Step 2: Building embeddings and saving to Milvus...")
    embedding_and_rag_builder(video_name, overwrite=overwrite)

    # Step 3: Run QA answer based on user query
    print("\n🤖 Step 3: Answering user query...")
    answer = qa_answering_builder(
        video_path=video_path,
        query=query,
        use_full_context=False,   # or True to test full context mode
        top_k=8,
        overwrite_embedding=False
    )

    print("\n💬 Final Answer:\n")
    print(answer)

if __name__ == "__main__":
    # === Replace with your test video and query ===
    test_videoqa_path = "input_videos/Lecture4.mp4"
    test_question = "Why does the slide recommend using a stack abstraction instead of directly using arrays?"
    test_pipeline_on_video(test_videoqa_path, test_question, overwrite=False)
