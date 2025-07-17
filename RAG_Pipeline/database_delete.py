import os
from pymilvus import connections, utility
from dotenv import load_dotenv
from VideoQA_constants.data import MILVUS_HOST, MILVUS_PORT

load_dotenv()

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT,
    user=os.getenv("MILVUS_USER"),
    password=os.getenv("MILVUS_PASSWORD"),
    db_name=os.getenv("MILVUS_DB_NAME")
)

def delete_video_collection(video_name: str) -> bool:
    """
    Deletes the Milvus collection for a specific video.
    """
    collection_name = f"videoqa_{video_name}"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"🗑️ Deleted Milvus collection: {collection_name}")
        return True
    else:
        print(f"⚠️ Collection not found: {collection_name}")
        return False
