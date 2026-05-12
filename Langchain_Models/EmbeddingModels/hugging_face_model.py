from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

document = [
    "Virat Kohli is an Indian cricketer and former captain of the Indian national team.",
    "MS Dhoni is a wicket-keeper batsman who has represented India in international cricket.",
    "Rohit Sharma is an Indian cricketer and the current captain of the Indian cricket team.",
    "Jasprit Bumrah is an Indian cricketer and a fast bowler who has represented India in international cricket."
]

vectors = embedding.embed_documents(document)

query = "Who is the captain of the Indian cricket team?"
query_vector = embedding.embed_query(query)

similarities = cosine_similarity([query_vector], vectors)

print(similarities)