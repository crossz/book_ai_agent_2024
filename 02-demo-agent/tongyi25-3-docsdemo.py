from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain.schema import HumanMessage
# Load environment variables
load_dotenv()

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


llm = ChatTongyi(model="qwen2.5-7b-instruct-1m")

llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke("What's 5 times forty two")
print(msg)
'''
为什么 msg 不直接包含结果？

在标准的 LangChain 工作流中，invoke 方法只生成模型的响应（包括工具调用），但不自动执行工具。执行工具通常需要额外的步骤，例如使用 ToolExecutor 或手动调用工具函数。这段代码只展示了模型的调用和输出，因此 msg 包含的是工具调用的描述，而不是计算后的结果。
如果代码中有额外的工具执行逻辑（比如一个代理或工具执行器），结果可能会被包含在 msg 的其他属性中（如 tool_responses），但当前代码没有这样的机制。
'''
# result = multiply(**msg.tool_calls[0]['args'])
result = multiply.invoke(msg.tool_calls[0]['args'])
print(result)


