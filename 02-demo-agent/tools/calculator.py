from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import operator

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression like '2 + 2' or '10 * 5'")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "A simple calculator for basic math operations (+, -, *, /)"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        # Remove all spaces from the expression
        expression = expression.replace(" ", "")
        
        # Define the operations
        operations = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv
        }
        
        # Find which operation to use
        op = None
        for symbol in operations.keys():
            if symbol in expression:
                op = symbol
                break
        
        if op is None:
            return "Error: Please use one of these operations: +, -, *, /"
            
        # Split the expression into two numbers
        try:
            num1, num2 = map(float, expression.split(op))
            result = operations[op](num1, num2)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: Could not perform calculation. Please use format like '2 + 2'"

    async def _arun(self, expression: str) -> str:
        return self._run(expression) 