from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("我想去 <{topic}> 旅行，我想知道这个地方有什么好玩的")

# 提示词模板
chat = ChatOpenAI(
    openai_api_key="sk-gRbZ9FJz2E7c7mwO5JOvp2u2rtoWoAbg12CxDy3Y25eLeDvd",
    openai_api_base="https://api.chatanywhere.tech",
    temperature=.7
                 )

# 输出模板
print(prompt.messages[0].prompt.input_variables)

output_parser = StrOutputParser()

chain = prompt | chat | output_parser

print(chain.invoke({"topic": "广州"}))
    