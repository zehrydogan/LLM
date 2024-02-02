from qdrant_client import QdrantClient
import pandas as pd

client = QdrantClient(host="localhost", port=6333)
client = QdrantClient(":memory:")

data = pd.read_csv("Text_Similarity_Dataset.csv")
data = data[["text1", "text2"]]
data = data.dropna()

client.add(
    collection_name="collection",
    documents=data.head(1000)['text1'].tolist() + data.head(1000)['text2'].tolist(),
)

search_result = client.query(
    collection_name="collection",
    query_text="parliament",
    limit=1,
    score_threshold=0.85
)

print('Result:', search_result[0].document, 'Score:', search_result[0].score)