!pip install transformers
!pip install sentence-transformers
!pip install pandas

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("/content/Text_Similarity_Dataset.csv")

text_columns = ["text1", "text2"]
data = data[text_columns]

data = data.dropna()

print(data.head())

model = SentenceTransformer('bert-base-nli-mean-tokens')

encoded_texts = model.encode(data["text1"].tolist() + data["text2"].tolist(), show_progress_bar=True)

encoded_text1 = encoded_texts[:len(data)]
encoded_text2 = encoded_texts[len(data):]

similarities = cosine_similarity(encoded_text1, encoded_text2)

print(similarities)