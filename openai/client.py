# 创建openai的client
from openai import OpenAI

client = OpenAI(
    api_key="sk-gRbZ9FJz2E7c7mwO5JOvp2u2rtoWoAbg12CxDy3Y25eLeDvd",
    base_url="https://api.chatanywhere.tech"
                )

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    #{"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "你是gpt几？."}
  ]
)

print(completion.choices[0].message)