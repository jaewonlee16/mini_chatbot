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
        "당신은 삼성전자를 대표하는 유용한 챗봇입니다."
        "당신은 질문에 답하고 삼성 제품에 대한 자세한 정보를 제공할 것입니다."
        "전문적인 어조를 유지하세요."
        "\n\n대화 기록:\n{history}\n사용자: {input}\n챗봇:"
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
    print("안녕하세요! 삼성전자 챗봇입니다. 어떻게 도와드릴까요?")
    print("(대화를 종료하시려면 'exit'을 입력하세요.)\n")

    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["종료", "exit", "quit"]:
            print("챗봇: 대화해주셔서 감사합니다. 좋은 하루 보내세요!")
            break

        response = conversation.invoke(input=user_input)
        print(f"챗봇: {response}\n")

if __name__ == "__main__":
    samsung_chatbot()
