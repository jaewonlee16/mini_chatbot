from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnableParallel

import prompts
from utils import debug_logging, print_context

# For OpenAI API
import os
from apikeys import OPENAI_API_KEY

import langchain
#langchain.verbose = True


import logging

# Step 0: Environment settings
# Set up OpenAI API key
#OPENAI_API_KEY = "your_openai_api_key_here"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Logging
# FILE:%(filename)s FUNC:%(funcName)s 
logging.basicConfig(
    format='[LINE%(lineno)d] %(levelname)s:%(message)s',
    level=logging.ERROR
)


# Step 1: Load documents from the directory
print("Loading documents...")
document_loader = DirectoryLoader(
    "./texts",  # Replace with the path to your directory
    glob="*.txt"  # Load only text files
)
docs = document_loader.load()

# Step 2: Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# Step 3: Embedding
embeddings = OpenAIEmbeddings()

# Step 4: Create DB
print("Creating vector store...")
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# Step 5: Retriever
retriever = vectorstore.as_retriever()

# Step 6: Define a template for the chatbot's response
prompt = PromptTemplate.from_template(prompts.korean_prompt)

# Step 7: LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 8: Chain
chain = (
    RunnableParallel({
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    )
    | RunnableLambda(print_context)
    | prompt
    | llm
    | StrOutputParser()
)


# Dictionary to store session history
session_history = {}

def get_session_history(session_ids):
    #print(f"[Session ID]: {session_ids}")
    if session_ids not in session_history:
        # Store new ChatMessageHistory in 'session_history'
        session_history[session_ids] = ChatMessageHistory()
    return session_history[session_ids]


# RAG Chain with history
rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  
    input_messages_key="question",  
    history_messages_key="chat_history",  
)


def samsung_chatbot(log_level = logging.INFO):
    print("안녕하세요! 삼성전자 챗봇입니다. 어떻게 도와드릴까요?")
    print("(대화를 종료하시려면 '종료'를 입력하세요.)\n")

    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["종료", "exit", "quit"]:
            print("챗봇: 대화해주셔서 감사합니다. 좋은 하루 보내세요!")
            break

        with debug_logging(log_level):
            response = rag_with_history.invoke({"question": user_input},
                                                 config={"configurable": {"session_id": "foo"}}
                                            )
        print(f"챗봇: {response}\n")


if __name__ == "__main__":
    log_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    print("Choose logging Level from 0 to 4 (2: logging.INFO shows retrieval context)")
    print("0: logging.DEBUG\n1: logging.INFO\n2: logging.WARNING \n3: logging.ERROR\n4: logging.CRITICAL")
    level = int(input())
    samsung_chatbot(log_levels[level])
