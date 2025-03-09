from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

chatLLM = ChatTongyi(
    model="qwen2.5-7b-instruct-1m",   # 此处以qwen-max为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    streaming=True,
    # other params...
)
res = chatLLM.stream([HumanMessage(content="阿里百炼免费的模型有哪些？")], streaming=True)
for r in res:
    print("chat resp:", r.content)
