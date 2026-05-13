from langchain.prompts import PromptTemplate
template = PromptTemplate(
    template="""You are a helpful assistant. Please provide a {answer_length} answer to the following question about 
    the research paper '{paper}': {question_type}""",
    input_variables=["paper", "question_type", "answer_length"],
    validate_template=True
)

template.save('template.json')