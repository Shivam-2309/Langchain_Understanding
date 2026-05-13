from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7)

st.header("Chat with Research Papers")

paper_input = st.selectbox("Select a research paper:", ["Paper 1: AI in Healthcare", "Paper 2: Climate Change Impacts", "Paper 3: Quantum Computing Advances"])
question_type_input = st.selectbox("Select a question type:", ["Summary", "Key Findings", "Methodology", "Future Work"])
answer_length_input = st.selectbox("Length of the answer:", ["Short (1-2 sentences)", "Medium (3-5 sentences)", "Detailed (6+ sentences)"])
# This is a static prompt for demonstration purposes. In a real application, you would want to dynamically generate this prompt 
# based on the user's question and the context of the research papers.
# user_input = st.text_input("Ask a question about research papers:")



# this is i am not giving the user the ability to provide the prompt directly, 
# instead we are generating the prompt based on the user's selections from the dropdowns. 
# This way we can ensure that the prompt is structured correctly and contains all the necessary 
# information for the model to generate a relevant response.
template = load_prompt = PromptTemplate.load('template.json')

prompt = template.invoke({
    'paper': paper_input,
    'question_type': question_type_input,
    'answer_length': answer_length_input
})


if st.button("Submit"):
    if prompt:
        result = model.invoke(prompt)
        st.write(result.content)
    else:
        st.write("Please enter a question.")
