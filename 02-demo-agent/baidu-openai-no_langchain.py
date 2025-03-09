from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()  

client = OpenAI(
    base_url='https://qianfan.baidubce.com/v2',
    api_key=os.getenv("BAIDU_APIKEY")
)
response = client.chat.completions.create(
    model="deepseek-r1-distill-qwen-7b", 
    messages=[
    {
        "role": "user",
        "content": "你可以提供哪些服务？"
    }
]
)
print(response)