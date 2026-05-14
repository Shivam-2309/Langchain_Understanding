from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# This pydantic parser is used to parse with unstructred response models. 
# however it will still work with structured response models as well.

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=100, # means the maximum number of tokens to generate in the output
    timeout=120 # means the maximum time to wait for a response from the model in seconds
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City of the person")

parser = PydanticOutputParser(pydantic_object=Person) 
# PydanticOutputParser is a class that takes a Pydantic model as input and provides methods to parse 
# the output of a language model into an instance of that Pydantic model. 
# In this case, we are using the Person model defined above, which has three fields: name, age, and city. 
# The parser will ensure that the output from the language model adheres to the structure defined by the Person model.

template = PromptTemplate(
    template="""
            Generate a fictional {place} person.

            {format_instructions}
            """,
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

print("CHECK: ", template.format(place="Indian"))

formatted_prompt = template.format(place="Indian")

result = model.invoke(formatted_prompt)

print("CHECKx: ", result)

print("CHECK1: ", result.content)
print("CHECK1 type: ", type(result.content))

final_result = parser.parse(result.content)

print("CHECK2: ", final_result)
print("CHECK2 type: ", type(final_result))

# chain = template | model | parser
# final_result = chain.invoke({'place':"Sri Lankan"})

# print("CHECk3: ", final_result)