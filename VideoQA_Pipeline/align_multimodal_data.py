import os
import json
from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus
from VideoQA_constants.prompts import RAG_PROMPT_TEMPLATE_NORMAL_ACTION, RAG_PROMPT_TEMPLATE_QA_GENERATE

def format_chunks(chunks):
    """
    Convert retrieved chunks into readable prompt format.
    Format: [audio] 10.42–16.38s: text...
            [visual] 297–435s: text...
    """
    lines = []
    for entry in chunks:
        source = entry.get("source", "unknown")
        timestamp = entry.get("timestamp", "")
        text = entry.get("text", "")
        lines.append(f"[{source}] {timestamp}: {text}")
    return "\n".join(lines)

def load_all_chunks(video_name):
    """
    Load and format merged visual/audio chunk data from JSONL file.
    Returns a formatted string for prompt injection.
    """
    chunk_path = os.path.join("outputs", "chunks", video_name, "all_chunks.jsonl")
    if not os.path.exists(chunk_path):
        return "No retrieved multimodal segments available."

    lines = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            source = entry.get("source", "unknown")
            timestamp = entry.get("timestamp", "")
            text = entry.get("text", "")
            lines.append(f"[{source}] {timestamp}: {text}")

    return "\n".join(lines)

def align_multimodal_data(multimodal_context, user_question, use_full_context=False, top_k=50):
    """
    Build final prompt using either full-context or top-k retrieved chunks from Milvus.
    If Milvus collection is empty, automatically build it from all_chunks.jsonl.
    """
    video_name = multimodal_context["video_name"]
    video_type = multimodal_context["video_type"]
    video_duration = multimodal_context["video_duration"]

    if use_full_context:
        retrieved_context = load_all_chunks(video_name)
    else:
        retriever = RagRetrieverMilvus(video_name)
        if retriever.collection.num_entities == 0:
            print(f"⚠️ Milvus collection for '{video_name}' is empty. Building from chunks...")
            retriever.build_from_chunks()
        top_chunks = retriever.query(user_question, top_k=top_k)
        retrieved_context = format_chunks(top_chunks)

    # prompt_template = RAG_PROMPT_TEMPLATE_NORMAL_ACTION
    prompt_template = RAG_PROMPT_TEMPLATE_QA_GENERATE
    return prompt_template.format(
        video_duration=video_duration,
        retrieved_context=retrieved_context,
        user_question=user_question
    )
