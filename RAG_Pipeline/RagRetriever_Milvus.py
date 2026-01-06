import os, json, uuid, openai
from typing import List
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from RAG_Pipeline.database_delete import delete_video_collection

load_dotenv()
openai.api_key  = os.getenv("LITELLM_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE")

# Embedding
EMBEDDING_MODEL   = "text-embedding-3-large"
DEFAULT_DIM       = 3072
EMBEDDING_OUT_DIM = int(os.getenv("EMBEDDING_OUT_DIM", str(DEFAULT_DIM)))

# Milvus
MILVUS_URI   = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_HOST  = os.getenv("MILVUS_HOST")
MILVUS_PORT  = os.getenv("MILVUS_PORT", "19530")
MILVUS_USER  = os.getenv("MILVUS_USER") or ""
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD") or ""
MILVUS_DB_NAME  = os.getenv("MILVUS_DB_NAME") or "default"

# Index/Search params
MILVUS_METRIC     = (os.getenv("MILVUS_METRIC") or "COSINE").upper()
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
MILVUS_NLIST      = int(os.getenv("MILVUS_NLIST", "128"))
MILVUS_NPROBE     = int(os.getenv("MILVUS_NPROBE", "10"))

# Limits/batching
TEXT_MAX_VARCHAR  = int(os.getenv("MILVUS_TEXT_MAX_LENGTH", "32768"))
EMBED_BATCH_SIZE  = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBED_MAX_CHARS   = int(os.getenv("EMBED_MAX_CHARS", "8000"))
INSERT_BATCH_ROWS = int(os.getenv("MILVUS_INSERT_BATCH_ROWS", "2000"))

class RagRetrieverMilvus:
    def __init__(self, video_name: str, overwrite: bool = False):
        self.video_name = video_name
        self.collection_name = f"videoqa_{video_name}"
        self._connect()
        self._ensure_collection(overwrite)

    def _connect(self):
        if MILVUS_URI:
            if MILVUS_TOKEN:
                connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN, db_name=MILVUS_DB_NAME)
            else:
                connections.connect(alias="default", uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD, db_name=MILVUS_DB_NAME)
        else:
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT,
                                user=MILVUS_USER, password=MILVUS_PASSWORD, db_name=MILVUS_DB_NAME)

    def _collection_compatible(self) -> bool:
        try:
            col = Collection(self.collection_name)
            for f in col.schema.fields:
                if f.name == "embedding":
                    dim = getattr(f, "params", {}).get("dim", None)
                    return (dim is None) or (int(dim) == int(EMBEDDING_OUT_DIM))
        except Exception:
            return False
        return True

    def _ensure_collection(self, overwrite: bool = False):
        if utility.has_collection(self.collection_name):
            if overwrite or not self._collection_compatible():
                delete_video_collection(self.video_name)
            else:
                self.collection = Collection(self.collection_name); self.collection.load(); return

        print(f"🛠️ Creating Milvus collection: {self.collection_name}")
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_OUT_DIM),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=TEXT_MAX_VARCHAR),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=20),
        ]
        schema = CollectionSchema(fields=fields, description=f"Multimodal chunks for {self.video_name}")
        collection = Collection(name=self.collection_name, schema=schema)
        collection.create_index(field_name="embedding", index_params={
            "metric_type": MILVUS_METRIC, "index_type": MILVUS_INDEX_TYPE, "params": {"nlist": MILVUS_NLIST}
        })
        print(f"✅ Index created on '{self.collection_name}' (metric={MILVUS_METRIC}, index={MILVUS_INDEX_TYPE})")
        self.collection = Collection(self.collection_name); self.collection.load()

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        clean = []
        for t in texts:
            t = "" if t is None else str(t)
            t = t.replace("\u0000", " ").strip()
            if len(t) > EMBED_MAX_CHARS: t = t[:EMBED_MAX_CHARS]
            clean.append(t)

        vecs: List[List[float]] = []
        for i in range(0, len(clean), EMBED_BATCH_SIZE):
            batch = clean[i:i+EMBED_BATCH_SIZE]
            kwargs = {}
            if EMBEDDING_OUT_DIM and EMBEDDING_OUT_DIM != 3072: kwargs["dimensions"] = EMBEDDING_OUT_DIM
            resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=batch, **kwargs)
            vecs.extend([d.embedding for d in resp.data])
        return vecs

    def build_from_chunks(self):
        path = os.path.join("outputs", "chunks", self.video_name, "all_chunks.jsonl")
        if not os.path.exists(path):
            print(f"❌ Chunk file not found: {path}"); return

        texts, ids, ts, src = [], [], [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                e = json.loads(line)
                text = e.get("text", "")
                if not str(text).strip(): continue
                if len(text) > TEXT_MAX_VARCHAR: text = text[:TEXT_MAX_VARCHAR]
                texts.append(text)
                ids.append(str(uuid.uuid4()))
                ts.append(e.get("timestamp", "")); src.append(e.get("source","unknown"))

        total = len(texts)
        print(f"🧠 Embedding {total} chunks for `{self.video_name}`...")
        vecs = self._embed_texts(texts)

        print(f"📥 Inserting into `{self.collection_name}` in batches of {INSERT_BATCH_ROWS} rows ...")
        for i in range(0, total, INSERT_BATCH_ROWS):
            j = min(i+INSERT_BATCH_ROWS, total)
            self.collection.insert([ids[i:j], vecs[i:j], texts[i:j], ts[i:j], src[i:j]])
        self.collection.flush()
        print(f"✅ Inserted {total} entries to `{self.collection_name}`")

    def query(self, question: str, top_k: int = 100) -> List[dict]:
        emb = self._embed_texts([question])[0]
        res = self.collection.search(
            data=[emb], anns_field="embedding",
            param={"metric_type": MILVUS_METRIC, "params": {"nprobe": MILVUS_NPROBE}},
            limit=top_k, output_fields=["text","timestamp","source"]
        )
        hits = res[0]
        return [{"text":h.entity.get("text"),"timestamp":h.entity.get("timestamp"),"source":h.entity.get("source")} for h in hits]
