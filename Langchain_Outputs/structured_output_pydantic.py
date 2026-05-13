# pydantic is a data validation library which is used to validate the output of the model and ensure that it is in the correct format. 
# It also provides better error messages when the output is not in the correct format.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel

class Review(BaseModel):
    name: str

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
structured_model = model.with_structured_output(Review)

movie = input("Enter the name of the movie you want to review: ")
result = structured_model.invoke(f"Write a review for the movie {movie} in 5 lines.")