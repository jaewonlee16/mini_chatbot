from langchain import PromptTemplate, OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from apikeys import OPENAI_API_KEY

import langchain
langchain.verbose = True

# Set up OpenAI API key
#OPENAI_API_KEY = "your_openai_api_key_here"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define a template for the chatbot's response
chatbot_template = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        "You are a helpful chatbot representing SAMSUNG Electronics. "
        "You will answer questions and provide detailed information about Samsung products. "
        "Maintain a professional and engaging tone. "
        "\n\nConversation History:\n{history}\nUser: {input}\nChatbot:"
    ),
)

# Initialize the language model and memory
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
memory = ConversationBufferMemory()

# Set up the conversation chain
conversation = ConversationChain(
    llm=llm,
    prompt=chatbot_template,
    memory=memory,
)

def samsung_chatbot():
    print("Hello! I'm your Samsung Electronics chatbot. How can I assist you today?")
    print("(Type 'exit' to end the conversation.)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Thank you for chatting with me! Have a great day!")
            break

        response = conversation.invoke(input=user_input)
        print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    samsung_chatbot()
