"""
Example 08: streaming responses.

By default, `chat.completions.create` waits until the *entire* answer is ready,
then hands it to you in one piece. With `stream=True`, the API instead sends the
answer back in small pieces ("chunks") as the model generates them, exactly like
you see text appear word-by-word in ChatGPT.

Why stream?
  - Perceived speed: the user sees the first words almost immediately instead of
    staring at a blank screen.
  - Long answers: you can start processing/displaying before it's finished.

How it works:
  - The call returns an *iterator* instead of a single response object.
  - Each chunk carries a `delta`: the new bit of content since the last chunk.
  - `delta.content` is often an empty string (e.g. the very first chunk just
    opens the message), so we guard for None/empty.
  - To get token `usage` while streaming, you must opt in with
    `stream_options={"include_usage": True}`; it arrives in the final chunk.

Run it:

    secrun python examples/08_streaming.py
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

stream = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "Write a haiku about streaming data."}],
    stream=True,
    stream_options={"include_usage": True},
    extra_body=extra_params
)

usage = None
for chunk in stream:
    # The last chunk carries usage but no choices, so check before indexing.
    if chunk.usage is not None:
        usage = chunk.usage
    if chunk.choices:
        piece = chunk.choices[0].delta.content
        if piece:
            # end="" + flush so the text appears live instead of line-buffered.
            print(piece, end="", flush=True)

print("\n\n--- usage ---")
print(usage)
