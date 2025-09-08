import os
from pymilvus import connections, utility
from dotenv import load_dotenv
from VideoQA_constants.data import MILVUS_HOST, MILVUS_PORT

load_dotenv()

# 先连接到 Milvus
connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT,
    user=os.getenv("MILVUS_USER"),
    password=os.getenv("MILVUS_PASSWORD"),
    db_name=os.getenv("MILVUS_DB_NAME")
)

# 再列出当前数据库里的 collections
collections = utility.list_collections()
print("📂 Collections in current DB:", collections)
