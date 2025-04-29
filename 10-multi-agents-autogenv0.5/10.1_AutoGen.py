# 导入autogen包
import autogen
import asyncio
import os
from dotenv import load_dotenv
import logging
from typing import List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 配置大模型
llm_config = {
    "config_list": [
        {
            "model": "deepseek-v3-241226",
            "api_key": os.getenv("BAIDU_API_KEY"),
            "base_url": "https://qianfan.baidubce.com/v2"
        }
    ],
    "timeout": 120,
    "temperature": 0.3,
    "max_tokens": 2000
}

# 定义鲜花电商的运营任务
tasks = {
    "inventory": [
        "分析当前库存中各种鲜花的数量，生成库存报告，并指出哪些鲜花库存不足。",
        "基于历史销售数据，预测未来一个月的鲜花需求趋势。"
    ],
    "market_research": [
        "分析当前市场趋势，识别最受欢迎的鲜花种类及其原因。"
    ],
    "content_creation": [
        "根据提供的信息，撰写一篇吸引人的博客文章，介绍最受欢迎的鲜花及选购技巧。"
    ]
}

# 创建Agent角色配置
inventory_config = {
    "name": "库存管理专家",
    "system_message": "你是一位经验丰富的库存管理专家，擅长分析库存数据并提供优化建议。",
    "llm_config": llm_config,
    "max_consecutive_auto_reply": 3,
    "human_input_mode": "NEVER"
}

market_research_config = {
    "name": "市场分析师",
    "system_message": "你是一位专业的市场分析师，擅长分析市场趋势和消费者行为。",
    "llm_config": llm_config,
    "max_consecutive_auto_reply": 3,
    "human_input_mode": "NEVER"
}

content_creator_config = {
    "name": "内容创作专家",
    "system_message": "你是一位专业的内容创作者，擅长撰写引人入胜的文章。你的文章结构清晰，内容丰富。",
    "llm_config": llm_config,
    "max_consecutive_auto_reply": 3,
    "human_input_mode": "NEVER"
}

# 创建代理
inventory_agent = autogen.AssistantAgent(**inventory_config)
market_research_agent = autogen.AssistantAgent(**market_research_config)
content_creator_agent = autogen.AssistantAgent(**content_creator_config)

# 创建用户代理
user_proxy = autogen.UserProxyAgent(
    name="用户代理",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("任务完成"),
    code_execution_config={
        "work_dir": "tasks",
        "use_docker": False,
        "last_n_messages": 3,
        "timeout": 300
    },
    max_consecutive_auto_reply=3
)

async def run_chats():
    try:
        # 启动库存管理对话
        logger.info("启动库存管理对话...")
        await user_proxy.a_initiate_chat(
            inventory_agent,
            message=tasks["inventory"][0],
            clear_history=True,
            max_consecutive_auto_reply=3
        )

        # 启动市场研究对话
        logger.info("启动市场研究对话...")
        await user_proxy.a_initiate_chat(
            market_research_agent,
            message=tasks["market_research"][0],
            max_turns=3,
            max_consecutive_auto_reply=3
        )

        # 启动内容创作对话
        logger.info("启动内容创作对话...")
        await user_proxy.a_initiate_chat(
            content_creator_agent,
            message=tasks["content_creation"][0],
            carryover="请在文章中包含数据表格或图表以增强可读性。",
            max_consecutive_auto_reply=3
        )

    except Exception as e:
        logger.error(f"对话过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("开始执行多代理对话...")
        asyncio.run(run_chats())
        logger.info("所有对话已完成！")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
