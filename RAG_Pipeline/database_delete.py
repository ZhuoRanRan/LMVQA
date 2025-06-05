from pymilvus import connections, utility
from VideoQA_constants.data import MILVUS_HOST, MILVUS_PORT

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

def delete_video_collection(video_name: str) -> bool:
    """
    Deletes the Milvus collection for a specific video.
    
    Args:
        video_name (str): The name of the video (e.g., "Lecture1")

    Returns:
        bool: True if collection deleted, False if not found
    """
    collection_name = f"videoqa_{video_name}"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"🗑️ Deleted Milvus collection: {collection_name}")
        return True
    else:
        print(f"⚠️ Collection not found: {collection_name}")
        return False
