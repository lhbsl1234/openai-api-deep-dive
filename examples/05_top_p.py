"""
Example 05: top_p (nucleus sampling).

`top_p` is the other randomness knob. Instead of scaling probabilities like
temperature does, it *restricts the candidate pool*:

  top_p = 0.1  -> consider only the smallest set of tokens whose probabilities
                  add up to 10%. Very focused: picks from the obvious choices.
  top_p = 1.0  -> consider everything (no restriction). This is the default.

Mental model: temperature changes *how boldly* the model chooses among options;
top_p changes *how many options it's even allowed to consider*.

Important: OpenAI recommends tuning EITHER temperature OR top_p, not both at
once, because they interact in confusing ways. Pick one knob and learn it.

Run it:

    secrun python examples/05_top_p.py
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

prompt = "Name an unusual but real animal."

for p in (0.1, 1.0):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        top_p=p,
        extra_body=extra_params
        # We leave temperature at its default and only vary top_p here.
    )
    print(f"top_p={p:<4} -> {response.choices[0].message.content}")
