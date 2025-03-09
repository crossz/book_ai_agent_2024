from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tools.calculator import CalculatorTool
from dotenv import load_dotenv
import os

def setup_llm():
    # Load environment variables
    load_dotenv()
    
    # Initialize Tongyi (Qwen) Chat LLM
    llm = ChatTongyi(
        model="qwen2.5-7b-instruct-1m",
        temperature=0.7,
        streaming=True
    )
    return llm

def main():
    # Initialize LLM
    llm = setup_llm()

    # Initialize calculator tool
    calculator = CalculatorTool()
    tools = [calculator]

    # Initialize agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Example prompts to test the system
    test_prompts = [
        "What is 25 multiplied by 13?",
        "If I have 150 items and want to divide them equally among 6 people, how many items does each person get?",
        "What is the square root of 144 plus 50?"
    ]

    # Run test prompts
    for prompt in test_prompts:
        print(f"\nQuestion: {prompt}")
        response = agent.run(prompt)
        print(f"Answer: {response}")

if __name__ == "__main__":
    main() 