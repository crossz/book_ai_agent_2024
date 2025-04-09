# def main():
#     print("Hello from 08-llamaindex-rag-agent!")


# if __name__ == "__main__":
#     main()


import os
from dotenv import load_dotenv
load_dotenv()  

from openai import OpenAI
client = OpenAI(
    base_url='https://qianfan.baidubce.com/v2',
    api_key=os.getenv("BAIDU_API_KEY")
)
response = client.chat.completions.create(
    model="deepseek-v3-241226", 
    messages=[{"role":"user","content":"你是什么大模型"}], 
    temperature=0.8, 
    top_p=0.8
)
print(response)