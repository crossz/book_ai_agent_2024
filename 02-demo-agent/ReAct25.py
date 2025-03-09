from pydantic import BaseModel, Field
from langchain_core.outputs import LLMResult
from langchain_core.language_models.llms import LLM
import os
from typing import List, Optional
import requests

from dotenv import load_dotenv
load_dotenv()  

# 导入LangChain Hub
from langchain import hub
# 从hub中获取React的Prompt
prompt = hub.pull("hwchase17/react")
print(prompt)

# # 导入ChatOpenAI
# from langchain_community.llms import OpenAI
# # 选择要使用的LLM
# llm = OpenAI()





# 自定义 SiliconFlowLLM 类
class SiliconFlowLLM(LLM):
    model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct")  # 默认模型
    temperature: float = Field(default=0.7)         # 默认温度
    max_tokens: int = Field(default=512)            # 默认最大 token 数
    api_key: str = Field(default=None)              # API 密钥

    def __init__(self, **kwargs):
        # 从环境变量加载 API 密钥，如果未提供则抛出异常
        api_key = kwargs.get("api_key") or os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("请设置 SILICONFLOW_API_KEY 环境变量或在初始化时提供 api_key")
        super().__init__(api_key=api_key, **kwargs)

    def _call(self, prompt: str, stop: list = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7,
        }
        response = requests.post(self.api_endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("text", "Error: No response from SiliconFlow")

    @property
    def _llm_type(self) -> str:
        return "siliconflow"

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """生成文本的核心方法"""
        url = "https://api.siliconflow.cn/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 处理每个提示
        generations = []
        for prompt in prompts:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            if stop:
                payload["stop"] = stop

            # 发送请求
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # 如果请求失败抛出异常
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            generations.append([{"text": text}])

        return LLMResult(generations=generations)

# 实例化 SiliconFlowLLM
llm = SiliconFlowLLM()
# ------


# 导入SerpAPIWrapper即工具包
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
# 实例化SerpAPIWrapper
search = SerpAPIWrapper()
# 准备工具列表
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="当大模型没有相关知识时，用于搜索知识"
    ),
]

# 导入create_react_agent功能
from langchain.agents import create_react_agent
# 构建ReAct代理
agent = create_react_agent(llm, tools, prompt)

# 导入AgentExecutor
from langchain.agents import AgentExecutor
# 创建代理执行器并传入代理和工具
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 调用代理执行器，传入输入数据
print("第一次运行的结果：")
agent_executor.invoke({"input": "当前Agent最新研究进展是什么?"})
print("第二次运行的结果：")
agent_executor.invoke({"input": "当前Agent最新研究进展是什么?"})