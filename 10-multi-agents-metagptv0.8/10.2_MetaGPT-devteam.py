from metagpt.actions import Action, UserRequirement
import re

class CodeGeneration(Action):
    PROMPT_TEMPLATE:str = """
    Generate a Python file for the following requirement:
    {instruction}
    
    Requirements:
    1. Include complete class/function definitions.
    2. Provide at least two test cases.
    3. Return the code in markdown format with ```python.
    """
    name: str = "Code Generation by the frontend developer"

    async def run(self, instruction: str):
        # 调用LLM生成代码
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        response = await self._aask(prompt)  # 核心方法，调用LLM接口
        
        # 提取代码块
        code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)
        return code_blocks[0].strip() if code_blocks else response
    

from metagpt.roles import Role
from metagpt.schema import Message

class SoftwareEngineer(Role):
    name: str = "DevBot"
    profile: str = "Software Engineer"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([CodeGeneration])  # 绑定Action
        self._watch([UserRequirement])     # 监听用户需求消息类型
    
    async def _act(self) -> Message:
        # 获取最新用户需求
        latest_msg = self.get_memories(k=1)[0]
        
        # 执行Action生成代码
        code = await self.rc.todo.run(latest_msg.content)
        
        # 构造返回消息
        return Message(content=code, role=self.profile, cause_by=type(self.rc.todo))


import asyncio

async def main():
    # 初始化角色
    engineer = SoftwareEngineer()
    
    # 发送用户需求
    req = "Write a CLI-based calculator supporting add/subtract/multiply/divide."
    await engineer.run(req)
    
    # 查看生成结果
    print(engineer.rc.memory.get()[-1].content)

if __name__ == "__main__":
    asyncio.run(main())

