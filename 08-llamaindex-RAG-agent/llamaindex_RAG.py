# 加载电商财报数据
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 

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
from llama_index.llms.openai import OpenAI
# llm = OpenAI(model="gpt-3.5-turbo-0613")

from llama_index.llms.openai_like import OpenAILike
# llm = OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"), 
#             base_url="https://api.siliconflow.cn/v1")

llm_s = OpenAILike(
    api_base="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    # model="Qwen/Qwen2.5-32B-Instruct", # works
    model="Qwen/Qwen2.5-7B-Instruct", # reactagent is not smart enough to 进行复杂分析 
    is_chat_model=True,  # Required for chat completions
    timeout=60  # Adjust if encountering timeout errors
)
llm_b = OpenAILike(
    api_base='https://qianfan.baidubce.com/v2',
    api_key=os.getenv("BAIDU_API_KEY"),
    # model="Qwen/Qwen2.5-32B-Instruct", # works
    model="deepseek-v3-241226", # reactagent is not smart enough to 进行复杂分析 
    is_chat_model=True,  # Required for chat completions
    timeout=60  # Adjust if encountering timeout errors
)
llm = llm_b
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


# # customize prompt for reactagent
from llama_index.core import PromptTemplate
from llama_index.core.prompts import RichPromptTemplate

custom_react_prompt2 = PromptTemplate(
    """
    你可以用的工具有： {tool_desc}

    ## Output Format
    以尽量口语化的、幽默搞笑的方式展示这个分析内容

    ```
    Thought: [根据用户的问题，分析出下一步的行动]
    Action: <根据思考的内容选择的工具>
    Action Input: <Input to the tool in JSON format>
    Observation: <工具调用后得到的结果>
    ```

    After collecting all necessary data, provide the final answer.

    """
)

custom_react_prompt = PromptTemplate(
    """
    You are an intelligent financial analyst designed to compare the financial data of two companies. 
    Your task is to extract and compare financial data from the provided RAG system. 

    ## Tools
    You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
    This may require breaking the task into subtasks and using different tools to complete each subtask.

    You have access to the following tools:
    {tool_desc}

    ## Workflow
    1. **Extract Financial Data**: 
       - First, retrieve sufficient financial data for each company. Ensure you gather enough data to perform a meaningful comparison.
    
    2. **Compare Financial Data**:
       - After collecting all necessary financial data for both companies, analyze and compare the data.
       - Provide a structured comparison highlighting key differences and similarities.

    ## Output Format
    To answer the question, please use the following format:

    ```
    Thought: [Your thought process here]
    Action: [Tool name if using a tool]
    Action Input: [Input to the tool in JSON format]
    Observation: [Result of the tool]
    ```

    After collecting all necessary data, provide the final answer in the following format:

    ```
    Answer: 
    - Company A:
      - Revenue: [Value]
      - Profit: [Value]
      - Assets: [Value]
      - Liabilities: [Value]
    - Company B:
      - Revenue: [Value]
      - Profit: [Value]
      - Assets: [Value]
      - Liabilities: [Value]
    - Comparison:
      - [Key insights and differences between the two companies]
    ```

    ## Rules
    - Always start with a Thought.
    - Do not attempt to compare the companies until you have collected sufficient financial data for both.
    - Ensure the financial data is comprehensive enough to support a meaningful comparison.
    - Use valid JSON format for tool inputs.
    - If you cannot retrieve enough data, explain the limitations in your answer.

    ## Current Question
    Compare the financial data of Company A and Company B.
    """
)



# 创建ReAct Agent
from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)


# # 打印系统提示头
prompt_dict = agent.get_prompts()
# print("Original Think Prompt:\n", prompt_dict)
# print("Original Think Prompt:\n", prompt_dict["agent_worker:system_prompt"].template)

prompt_dict["agent_worker:system_prompt"].template = custom_react_prompt2.template
agent.update_prompts(prompt_dict)


# 让Agent完成任务
agent.chat("对比这两家公司的财务数据")
