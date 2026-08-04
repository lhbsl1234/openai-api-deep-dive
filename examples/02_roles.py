"""
Example 02: system / user / assistant roles.

A chat is a transcript of messages, each tagged with a role:

  - system    : Standing instructions. Sets the persona, rules, and tone for the
                whole conversation. Usually the first message. The user doesn't
                "see" it; it steers everything that follows.
  - user      : What the human says.
  - assistant : What the model said. You include PRIOR assistant messages to give
                the model memory of the conversation. The API itself is
                stateless, so *you* resend the history every time.

Run it:

    secrun python examples/02_roles.py

Try editing the system message (e.g. "You are a grumpy pirate") and watch the
tone of the answer change without touching the question at all. That's the power
of the system role.
"""
from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
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

# Note the assistant message in the middle: we're *simulating* a prior turn so
# the model continues the thread coherently. This is how you build multi-turn
# chat: keep appending messages to the list.
messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a terse math tutor. One line only."},
    {"role": "user", "content": "What is 12 * 12?"},
    {"role": "assistant", "content": "144."},
    {"role": "user", "content": "And that, doubled?"},
]

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    extra_body=extra_params
)

print(response.choices[0].message.content)
