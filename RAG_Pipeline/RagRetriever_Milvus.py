import os
import json
import uuid
import openai
from typing import List
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from dotenv import load_dotenv

from VideoQA_constants.data import MILVUS_HOST, MILVUS_PORT, EMBEDDING_DIM
from RAG_Pipeline.database_delete import delete_video_collection

load_dotenv()
openai.api_key = os.getenv("LITELLM_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE")


class RagRetrieverMilvus:
    def __init__(self, video_name: str, overwrite: bool = False):
        self.video_name = video_name
        self.collection_name = f"videoqa_{video_name}"

        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            user=os.getenv("MILVUS_USER"),
            password=os.getenv("MILVUS_PASSWORD"),
            db_name=os.getenv("MILVUS_DB_NAME")
        )

        self._ensure_collection(overwrite)

    def _ensure_collection(self, overwrite: bool = False):
        if utility.has_collection(self.collection_name):
            if overwrite:
                delete_video_collection(self.video_name)
            else:
                self.collection = Collection(self.collection_name)
                self.collection.load()
                return

        print(f"🛠️ Creating Milvus collection: {self.collection_name}")
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=20),
        ]
        schema = CollectionSchema(fields=fields, description=f"Multimodal chunks for {self.video_name}")
        collection = Collection(name=self.collection_name, schema=schema)

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"✅ Index created on '{self.collection_name}'")

        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [d.embedding for d in response.data]

    def build_from_chunks(self):
        chunk_path = os.path.join("outputs", "chunks", self.video_name, "all_chunks.jsonl")
        if not os.path.exists(chunk_path):
            print(f"❌ Chunk file not found: {chunk_path}")
            return

        texts, ids, timestamps, sources = [], [], [], []

        with open(chunk_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                text = entry.get("text", "")
                if not text.strip():
                    continue
                texts.append(text)
                ids.append(str(uuid.uuid4()))
                timestamps.append(entry.get("timestamp", ""))
                sources.append(entry.get("source", "unknown"))

        print(f"🧠 Embedding {len(texts)} chunks for `{self.video_name}`...")
        vectors = self._embed_texts(texts)

        data = [ids, vectors, texts, timestamps, sources]
        self.collection.insert(data)
        self.collection.flush()
        print(f"✅ Inserted {len(texts)} entries to `{self.collection_name}`")

    def query(self, question: str, top_k: int = 50) -> List[dict]:
        embedding = self._embed_texts([question])[0]
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text", "timestamp", "source"]
        )
        hits = results[0]
        return [
            {
                "text": hit.entity.get("text"),
                "timestamp": hit.entity.get("timestamp"),
                "source": hit.entity.get("source"),
            }
            for hit in hits
        ]
