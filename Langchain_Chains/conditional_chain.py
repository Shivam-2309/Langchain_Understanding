from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7)

parser1 = StrOutputParser()

class Feedback(BaseModel) :
    sentiment : Literal['Negative', 'Positive'] = Field(description="sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = "Give me the sentiment of this feedback: {feedback}, and classify it as positive or negative \n {format_instruction}", 
    input_variables=['feedback'], 
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# Now i have the sentiment now i have to decide the course of action to perfom 
# so i now need to have branch like structure to tell where my flow of code should go actually 

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser1),
    (lambda x:x.sentiment == 'Negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({'feedback' : "What a wonderful movie"})

print(result)
chain.get_graph().print_ascii()