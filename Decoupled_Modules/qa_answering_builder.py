import os
import openai
from dotenv import load_dotenv
from VideoQA_Pipeline.utils import get_video_duration, classify_video_type
from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus
from VideoQA_Pipeline.align_multimodal_data import align_multimodal_data

load_dotenv()
openai.api_key = os.getenv("LITELLM_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE")

def get_video_name(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]

def qa_answering_builder(
    video_path: str,
    query: str,
    use_full_context: bool = False,
    top_k: int = 50,
    overwrite_embedding: bool = False,
    model_name: str = "gpt-4o",
    max_tokens: int = 1500
) -> str:
    """
    Build prompt from multimodal chunks and return the model's answer to the query.

    Args:
        video_path (str): Path to the video.
        query (str): User question.
        use_full_context (bool): Use all chunks instead of retrieval.
        top_k (int): How many chunks to retrieve (if use_full_context=False).
        overwrite_embedding (bool): Force rebuilding embedding.
        model_name (str): OpenAI model name.
        max_tokens (int): Max tokens to generate.

    Returns:
        str: The model's answer.
    """
    video_name = get_video_name(video_path)
    video_duration = get_video_duration(video_path)

    frame_dir = os.path.join("outputs", "frames", f"{video_name}_frames")
    frame_files = [f for f in os.listdir(frame_dir) if f.endswith(".png")] if os.path.exists(frame_dir) else []
    video_type = classify_video_type(frame_files, video_duration)

    # Step 1: Ensure embeddings exist
    retriever = RagRetrieverMilvus(video_name, overwrite=overwrite_embedding)
    if retriever.collection.num_entities == 0:
        retriever.build_from_chunks()

    # Step 2: Construct context and prompt
    multimodal_context = {
        "video_name": video_name,
        "video_duration": video_duration,
        "video_type": video_type
    }

    prompt = align_multimodal_data(
        multimodal_context=multimodal_context,
        user_question=query,
        use_full_context=use_full_context,
        top_k=top_k
    )

    # Step 3: Query LLM
    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()
