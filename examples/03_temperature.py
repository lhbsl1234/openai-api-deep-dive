"""
Example 03: temperature.

`temperature` controls randomness, roughly 0.0 to 2.0.

  - 0.0  : The model almost always picks its single most likely next token.
           Answers are focused, repeatable, and a bit "safe". Best for facts,
           code, extraction, anything where you want consistency.
  - 0.7  : A balanced default. Some variety, still coherent.
  - 1.5+ : Wild. More surprising word choices, more risk of nonsense. Good for
           brainstorming or creative writing.

Run it:

    secrun python examples/03_temperature.py

We ask the same creative question at three temperatures and print the results
side by side. Notice how 0.0 tends to repeat itself across runs while the high
setting reinvents the answer each time.
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

prompt = "Give a five-word slogan for a coffee shop on the moon."

for temp in (0.0, 0.7, 1.5, 2.0):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    print(f"temperature={temp:<4} -> {response.choices[0].message.content}")
