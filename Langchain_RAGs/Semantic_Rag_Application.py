from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
loader = PyMuPDFLoader(
    "Bhagavad-gita-Swami-BG-Narasingha.pdf"
)

docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="gita_db"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)

question = input()
retrieved_docs = retriever.invoke(question)

print("Retrieved Documents:")
for i, doc in enumerate(retrieved_docs):
    print(f"Document {i+1}:")
    print(doc.page_content)
    print("\n---\n")

context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)

prompt = PromptTemplate(
    template="""
        You are acting as Lord Krishna from the Bhagavad Gita.

        Your task is to answer the user's question ONLY using the provided context from the Bhagavad Gita.

        Guidelines:
        - Speak with wisdom, compassion, clarity, and calmness like Lord Krishna.
        - Teach the user in a spiritual and philosophical manner.
        - Wherever appropriate, include relevant Bhagavad Gita verses or shlokas from the provided context.
        - Quote verses properly if they are available in the context.
        - Do NOT invent verses or hallucinate references.
        - If the answer is not present in the context, honestly say:
        "This teaching is not clearly present in the provided scripture context."
        - Keep the answer meaningful, practical, and around 200-300 words.
        - End the response with a short reflective spiritual takeaway for the user.

        Provided Context:
        {context}

        User Question:
        {question}

        Answer as Lord Krishna:
        """,
    input_variables=["context", "question"]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

chain = prompt | llm
response = chain.invoke({
    "context" : context, 
    "question" : question, 
})

print(response.content)