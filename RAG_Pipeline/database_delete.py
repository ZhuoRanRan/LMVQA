import os
from dotenv import load_dotenv
from pymilvus import connections, utility

load_dotenv()

MILVUS_URI   = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_HOST  = os.getenv("MILVUS_HOST")
MILVUS_PORT  = os.getenv("MILVUS_PORT", "19530")
MILVUS_USER  = os.getenv("MILVUS_USER") or ""
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD") or ""
MILVUS_DB_NAME  = os.getenv("MILVUS_DB_NAME") or "default"

def _connect():
    try:
        if connections.has_connection("default"): return
    except Exception:
        pass
    if MILVUS_URI:
        if MILVUS_TOKEN:
            connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN, db_name=MILVUS_DB_NAME)
        else:
            connections.connect(alias="default", uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD, db_name=MILVUS_DB_NAME)
    else:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, user=MILVUS_USER, password=MILVUS_PASSWORD, db_name=MILVUS_DB_NAME)

_connect()

def delete_video_collection(video_name: str) -> bool:
    name = f"videoqa_{video_name}"
    if utility.has_collection(name):
        utility.drop_collection(name)
        print(f"🗑️ Deleted Milvus collection: {name}"); return True
    else:
        print(f"⚠️ Collection not found: {name}"); return False

if __name__ == "__main__":
    print("📂 Collections in current DB:", utility.list_collections())

    # delete_video_collection("Lecture1")

    # for cname in utility.list_collections(): utility.drop_collection(cname)