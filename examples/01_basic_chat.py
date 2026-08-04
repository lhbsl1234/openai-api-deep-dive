"""
Example 01: Your first chat completion.

The whole API in five lines. You send a list of messages; you get back a
message. Run it:

    secrun python examples/01_basic_chat.py

What to notice:
  - `client = OpenAI()` reads your key from the OPENAI_API_KEY environment
    variable. We load it from .env first.
  - `messages` is a list. Even a one-off question is a list with one entry.
  - The reply lives at `response.choices[0].message.content`. There can be more
    than one choice if you ask for several (the `n` parameter), hence the [0].
  - `response.usage` reports exactly how many tokens you were billed for.
"""
from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# --------------------------
# 加载项目根目录下的 .env 文件
# --------------------------
# __file__ = examples/01_basic_chat.py
# parent.parent 往上跳一级到项目根目录
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
# print(f"Loaded .env from {env_path}")
# ======================
# 切换模型开关
# True → deepseek-v4-flash 云端
# False → 本地 qwen3:8b-q4_K_M (Ollama)
# ======================
# load_dotenv()
# if not os.getenv("OPENAI_API_KEY"):
#     sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

USE_DEEPSEEK = True
# USE_DEEPSEEK = False

client: OpenAI
model_name: str

if USE_DEEPSEEK:
    # 读取 DeepSeek Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not found in .env (see SECRETS.md), please check your configuration!")

    client = OpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=api_key
    )
    model_name = "deepseek-v4-flash"
    extra_params = {}   # 云端：不带任何ollama私有参数

else:
    # 本地 Ollama Qwen3
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="dummy"
    )
    model_name = "qwen3:8b-q4_K_M"
    extra_params = {"keep_alive": -1} # 本地Ollama才启用驻留
# client = OpenAI()


response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "In one sentence, what is flask?"},
    ],
    extra_body=extra_params
)

print(response.choices[0].message.content)
print("\n--- usage ---")
print(response.usage)
