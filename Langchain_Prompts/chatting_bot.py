from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7)


# This is a better way to interact as here i am also sending the chat history to the model
# chat_history = []
# while True:
#     user_input = input("You: ")
#     if user_input in ['exit', 'quit']:
#         print("Goodbye!")
#         break
#     chat_history.append({"role": "user", "content": user_input})
#     response = model.invoke(chat_history)
#     chat_history.append({"role": "assistant", "content": response.content})
#     print("Model: ", response.content)


# print("Chat ended.")
# print(chat_history)

# langchain provides a simpler and an elegant way to do all this 

chat_history = [SystemMessage(content="You are a helpful assistant.")]
while True:
    user_input = input("You: ")
    if user_input in ['exit', 'quit']:
        print("Goodbye!")
        break
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Model: ", response.content)


print("Chat ended.")
print(chat_history)