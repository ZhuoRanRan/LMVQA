import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# Azure gateway (OpenAI-compatible) example:
#   OPENAI_BASE_URL=https://<host>/openai/v1
#   OPENAI_API_KEY=...
#   OPENAI_DEPLOYMENT_NAME=o1-model
endpoint = os.environ["OPENAI_BASE_URL"]
deployment_name = os.environ["OPENAI_DEPLOYMENT_NAME"]
api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(base_url=endpoint, api_key=api_key)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message)