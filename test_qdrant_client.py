from qdrant_client import QdrantClient
client = QdrantClient(path="./qdrant_storage")
print("Client type:", type(client))
print("Available attributes:", [x for x in dir(client) if not x.startswith("_")])
