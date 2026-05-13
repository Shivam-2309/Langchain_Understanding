from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated # sometimes this fails to work properly, prefer pydantic 

load_dotenv()

class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the movie"] # this guides the model to understand real meaning of LLM
    title: Annotated[str, "The title of the movie"]
    rating: int
    actors: list[str]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
structured_model = model.with_structured_output(Review) # Behind the scenes, this line forces the model to return output in proper json format 

movie = input("Enter the name of the movie you want to review: ")
result = structured_model.invoke(f"Write a review for the movie {movie}.")

print(f"Title: {result['title']}")
print(f"Rating: {result['rating']}")
print("Actors:")
for actor in result['actors']:
    print(f"- {actor}")

# The above is a good way but it does not guarantee the correctness
# To ensure the correctness of the output, we can use pydantic for validation