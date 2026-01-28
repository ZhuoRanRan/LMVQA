from VideoQA_Pipeline.build_chunks import build_chunks
from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus

video_name = "Lecture1"

build_chunks(video_name)

retriever = RagRetrieverMilvus(video_name)
retriever.build_from_chunks()
