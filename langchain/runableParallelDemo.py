# 操作输入输出
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

vectorstore = faiss.FAISS.from_texts(
    texts=["harrison worked at kensho"], 
    embedding=OpenAIEmbeddings(
      openai_api_key="sk-gRbZ9FJz2E7c7mwO5JOvp2u2rtoWoAbg12CxDy3Y25eLeDvd",
      openai_api_base="https://api.chatanywhere.com.cn/v1",
    )
)
retriever = vectorstore.as_retriever()
print("retriever:",retriever)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


chat = ChatOpenAI(
  openai_api_key="sk-gRbZ9FJz2E7c7mwO5JOvp2u2rtoWoAbg12CxDy3Y25eLeDvd",
  openai_api_base="https://api.chatanywhere.tech",
  temperature=.7
)

retriever_chain = (
  {"context": retriever, "question": RunnablePassthrough()}
  | prompt
  | chat
  | StrOutputParser()
)

print(retriever_chain.invoke("where did harrison work?"))




