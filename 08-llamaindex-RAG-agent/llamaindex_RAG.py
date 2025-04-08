# 加载电商财报数据
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface  import HuggingFaceEmbedding 

import os
from dotenv import load_dotenv
load_dotenv()  

# 使用 BGE 小型英文模型
# settings = Settings(
#     embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# )
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model


# 配置大模型
# from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
# # llm = OpenAI(model="gpt-3.5-turbo-0613")
# llm = OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"), 
#             base_url="https://api.siliconflow.cn/v1")


llm = OpenAILike(
    api_base="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    # model="Qwen/Qwen2.5-32B-Instruct", # works
    model="Qwen/Qwen2.5-7B-Instruct", # reactagent is not smart enough to 进行复杂分析 
    is_chat_model=True,  # Required for chat completions
    timeout=60  # Adjust if encountering timeout errors
)
Settings.llm = llm

A_docs = SimpleDirectoryReader(
    input_files=["./电商A-Third Quarter 2023 Results.pdf"]
).load_data()
B_docs = SimpleDirectoryReader(
    input_files=["./电商B-Third Quarter 2023 Results.pdf"]
).load_data()



# 从文档中创建索引
from llama_index.core import VectorStoreIndex
A_index = VectorStoreIndex.from_documents(A_docs)
B_index = VectorStoreIndex.from_documents(B_docs)

# 持久化索引（保存到本地）
from llama_index.core import StorageContext
A_index.storage_context.persist(persist_dir="./storage/A")
B_index.storage_context.persist(persist_dir="./storage/B")


# 从本地读取索引
from llama_index.core import load_index_from_storage
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/A"
    )
    A_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/B"
    )
    B_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False


# 创建查询引擎
A_engine = A_index.as_query_engine(similarity_top_k=5)
B_engine = B_index.as_query_engine(similarity_top_k=5)


# 配置查询工具
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
query_engine_tools = [
    QueryEngineTool(
        query_engine=A_engine,
        metadata=ToolMetadata(
            name="A_Finance",
            description=(
                "用于提供A公司的财务信息 "
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=B_engine,
        metadata=ToolMetadata(
            name="B_Finance",
            description=(
                "用于提供B公司的财务信息 "
            ),
        ),
    ),
]






# 创建ReAct Agent
from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)


# 让Agent完成任务
agent.chat("对比所有公司的 Revenue，进行分析")
