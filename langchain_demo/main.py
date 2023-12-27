# %%
# 定义提示词模板
RESPONSE_TEPLATE = """\
你作为一名专业程序员和问题解决者，负责回答关于Langchain的任何问题并且严格遵守下面的8个准则。\
1.基于所提供搜索结果（URL和context）生成一份全面并且信息丰富的回答。\
2.你必须只使用所提供的搜索中的内容，将搜索结果合并成一个连贯的答案，不要重复内容，以公正无偏见的新闻报道风格撰写。\
3.使用[${{number}}]格式进行引用，只引用最相关的搜索结果来准确的回答问题。\
4.将引用放在引用放在引用它们的句子或段落末尾，而不是全部放在最后。\
5.如果不同的结果指的是同名的不同实体，为每个实体编写单独的答案。\
6.使用项目符号列表以提高可读性，将引用放在适当位置，而不是全部放在最后。\
7.如果上下文中没有与问题相关的信息，请说“嗯，我不确定。”不要试图编造答案。\
8.下面'context'的HTML块之间的任何内容都来自知识库，不是与用户的对话内容。\

<context>
    {context} 
<context/>

记住：如果上下文中没有相关信息，只需说“嗯，我不确定。”不要试图编造答案。位于上述'context'HTML块之间的任何内容都是从知识库中检索的，不是与用户的对话内容。\

"""

REPHRASE_TEMPLATE = """
鉴于以下对话和后续问题，请重新表述后续问题，使其成为一个独立的问题。\
历史对话:
{chat_history}
后续问题输入:
{question}
独立的问题:"""

# %%
# 设置环境变量
import os

os.environ['OPENAI_API_KEY'] = ""
os.environ['OPENAI_API_BASE'] = ""

# %%

# %%
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
# langchain服务端
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
# %%


# %%
from operator import itemgetter
from pydantic import BaseModel
from typing import Dict, List, Optional,Sequence
from langchain.schema.embeddings import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import Runnable,RunnableBranch,RunnableLambda,RunnableMap
from langchain.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from pydantic.dataclasses import dataclass


class Config:
    arbitrary_types_allowed = True



# 定义聊天类
class ChatRequest(BaseModel):
  """chat request model"""
  chat_history: Optional[List[Dict[str,str]]]
  question: str
  

# 定义向量解析模型
def get_embeddings_model() -> Embeddings:
  return OpenAIEmbeddings(
  openai_api_key=os.environ['OPENAI_API_KEY'],
  openai_api_base=os.environ['OPENAI_API_BASE']+ "/v1",
)

# 定义文档解析器
def get_retriever() -> BaseRetriever:
  els_client = ElasticsearchStore(
    es_url="http://154.204.60.125:9200",
    index_name="multi_index_1",
    embedding=get_embeddings_model(),
  )
  return els_client.as_retriever()

# 定义chain(定义路由是否依赖历史)
def create_retriever_chain(llm: BaseLanguageModel,retriever: BaseRetriever) -> Runnable:
  # 问题压缩
  CONDEN_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
  condense_question_chain = (
    CONDEN_QUESTION_PROMPT | llm | StrOutputParser()
  ).with_config(
    run_name="CondenseQuestion",
  )
  # 问题压缩与检索
  conversation_chain= condense_question_chain | retriever
  return RunnableBranch(
      (
          RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
          ),
          conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
      ),
      (
          RunnableLambda(itemgetter("question")).with_config(
            run_name="Itemgetter:question"
          )
          | retriever
      ).with_config(run_name="RetrievalChainWithNoHistory"),
  ).with_config(run_name="RouteDependingOnChatHistory")
  
# 格式化文档
def format_docs(docs: Sequence[Document]) -> str:
  formatted_docs = []
  for i,doc in enumerate(docs):
    doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
    formatted_docs.append(doc_string)
  return "\n".join(formatted_docs)

# 序列化历史记录
def serialize_history(request: ChatRequest):
  chat_history = request["chat_history"] or []
  converted_chat_history = []
  for message in chat_history:
    if message.get("human") is not None:
      converted_chat_history.append(HumanMessage(content=message["human"]))
    if message.get("ai") is not None:
      converted_chat_history.append(AIMessage(content=message["ai"]))
  return converted_chat_history

# 创建链函数
def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
  retriever_chain = create_retriever_chain(llm=llm,retriever=retriever).with_config(run_name="FindDocs")
  _context = RunnableMap(
    {
      "context": retriever_chain | format_docs,
      "question": itemgetter("question"),
      "chat_history": itemgetter("chat_history"),
    }
  ).with_config(run_name="RetrieveDocs")
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system",RESPONSE_TEPLATE),
      MessagesPlaceholder(variable_name="chat_history"),
      ("human","{question}")
    ]
  )

  #答案生成器
  response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
    run_name="GenerateResponse",
  )

  return (
    {
      "question": RunnableLambda(itemgetter("question")).with_config(
        run_name="Itemgetter:question"
      ),
      "chat_history": RunnableLambda(serialize_history).with_config(
        run_name="SerializeHistory"
      ),
    } | _context | response_synthesizer
  )

# 定义大语言模型
llm = ChatOpenAI(
  openai_api_key=os.environ['OPENAI_API_KEY'],
  openai_api_base=os.environ['OPENAI_API_BASE'],
)

# 获取文档解析器
retriever = get_retriever()

# 回答langchain
answer_chain = create_chain(
    llm,
    retriever,
)
  



# %%

add_routes(
  app,
  answer_chain,
  path="/chat",
  input_type=ChatRequest,
  config_keys=["metadata"]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


# %%
