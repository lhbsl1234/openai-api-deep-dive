"""
Example 04: max_tokens (and finish_reason).

`max_tokens` caps how many tokens the model is allowed to GENERATE. It does NOT
limit your input, and it does NOT make the model "summarize to fit". It simply
cuts the model off when the budget runs out, mid-sentence if necessary.

Why use it?
  - Cost control: output tokens are the expensive ones.
  - Latency: shorter answers come back faster.
  - Safety: stop a runaway answer from ballooning.

The companion to watch is `finish_reason`:
  - "stop"   : the model finished on its own.
  - "length" : it hit your max_tokens cap, so the answer is truncated.

Run it:

    secrun python examples/04_max_tokens.py
"""
from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

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

prompt = "Explain how the internet works."

# If you'd like to see the reason be "stop", add a large number like 2000 to
# the loop. But remember larger numbers incur more cost.
for cap in (16, 256):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cap,
        extra_body=extra_params
    )
    choice = response.choices[0]
    print(f"--- max_tokens={cap} (finish_reason={choice.finish_reason}) ---")
    print(choice.message.content)
    print()
