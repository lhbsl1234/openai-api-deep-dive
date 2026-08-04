"""
Example 06: stop sequences.

`stop` is a string (or list of up to 4 strings) that tells the model: "the
moment you're about to produce this text, stop generating." The stop text itself
is NOT included in the output.

Uses:
  - Cut a list off after N items (stop at "4.").
  - End a structured response at a delimiter.
  - Prevent the model from running past a known boundary (e.g. "\n\n").

Run it:

    secrun python examples/06_stop_sequences.py

The first call lets the model count freely; the second stops it the instant it
tries to write "4", so you only get items 1–3.
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

prompt = "Count from 1 to 10, one number per line, like '1.', '2.', ..."

print("--- without stop ---")
r1 = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    extra_body=extra_params
)
print(r1.choices[0].message.content)

print("\n--- with stop=['4.'] ---")
r2 = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    stop=["4."],
    extra_body=extra_params
)
print(r2.choices[0].message.content)
print(f"(finish_reason={r2.choices[0].finish_reason})")
