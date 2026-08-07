"""
Cost estimation for OpenAI chat models.

OpenAI bills you per *token*, and charges a different rate for tokens you send
(input / "prompt" tokens) versus tokens the model generates back (output /
"completion" tokens). Output tokens are usually several times more expensive
than input tokens, which is why a chatty model can cost more than you expect.

Prices are quoted per 1,000,000 tokens. We store them that way below and divide
when we estimate.

PRICES CHANGE. The numbers below are a snapshot and may be out of date by the
    time you read this. Always confirm against the official pricing page:
        https://platform.openai.com/docs/pricing
    Treat this module as a *teaching tool*, not a billing source of truth.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Price in US dollars per 1,000,000 tokens."""
    input_per_1m: float
    output_per_1m: float


# A small, representative slice of the catalog. Add more as you explore.
# (input $/1M, output $/1M)
PRICING: dict[str, ModelPrice] = {
    # ========== OpenAI ==========
    "gpt-4o":          ModelPrice(input_per_1m=2.50, output_per_1m=10.00),
    "gpt-4o-mini":     ModelPrice(input_per_1m=0.15, output_per_1m=0.60),
    "gpt-4-turbo":     ModelPrice(input_per_1m=10.00, output_per_1m=30.00),
    "gpt-3.5-turbo":   ModelPrice(input_per_1m=0.50, output_per_1m=1.50),
    "o1-preview":      ModelPrice(input_per_1m=15.00, output_per_1m=60.00),
    "o1-mini":         ModelPrice(input_per_1m=3.00, output_per_1m=12.00),

    # ========== DeepSeek 系列（官方国际API基准价 USD/1M tokens） ==========
    "deepseek-v4-flash":    ModelPrice(input_per_1m=0.14, output_per_1m=0.28),  # 原 deepseek-chat
    "deepseek-v4-pro":      ModelPrice(input_per_1m=0.42, output_per_1m=0.84),
    "deepseek-v3":          ModelPrice(input_per_1m=0.27, output_per_1m=1.10),
    "deepseek-r1":          ModelPrice(input_per_1m=0.55, output_per_1m=2.20),  # 推理模型
    "deepseek-coder-v2":    ModelPrice(input_per_1m=0.14, output_per_1m=0.28),

    # ========== Qwen / 通义千问系列（国际版标准价 USD/1M tokens） ==========
    "qwen3-max":            ModelPrice(input_per_1m=1.80, output_per_1m=6.00),
    "qwen3-plus":           ModelPrice(input_per_1m=0.26, output_per_1m=0.78),
    "qwen3-flash":          ModelPrice(input_per_1m=0.04, output_per_1m=0.16),
    "qwen3.7-plus":         ModelPrice(input_per_1m=0.80, output_per_1m=3.20),   # 新增
    "qwen3.8":              ModelPrice(input_per_1m=2.00, output_per_1m=6.00),   # 简称别名
    "qwen3.8-max":          ModelPrice(input_per_1m=2.00, output_per_1m=6.00),   # 官方标准ID
    "qwen2.5-72b-instruct": ModelPrice(input_per_1m=0.35, output_per_1m=0.40),
    "qwen2.5-14b-instruct": ModelPrice(input_per_1m=0.10, output_per_1m=0.05),
    "qwen2.5-7b-instruct":  ModelPrice(input_per_1m=0.01, output_per_1m=0.01),
}

# Embedding models (see examples/11_embeddings.py) are billed differently: there
# is no "output" to generate, so you only pay for the tokens you send in. We keep
# them in their own table with a single price per 1M tokens.
# Price unit: USD per 1M input tokens
EMBEDDING_PRICING: dict[str, float] = {
    # ========== OpenAI Embedding ==========
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,

    # ========== Qwen / 通义千问 Embedding（DashScope 国际版） ==========
    "text-embedding-v2":    0.12,
    "text-embedding-v3":    0.09,
    "text-embedding-v4":    0.084,
    "qwen3.7-text-embedding": 0.075,

    # ========== DeepSeek Embedding（DeepSeek 官方国际API） ==========
    "deepseek-embedding":   0.002,

    # ========== 其他主流国产商用 Embedding 常用API名称 ==========
    "bge-m3":               0.010,
    "bge-large-zh-v1.5":    0.012,
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the estimated cost in USD for a request/response.

    Raises KeyError with a helpful message if we don't know the model's price.
    """
    if model not in PRICING:
        known = ", ".join(sorted(PRICING))
        raise KeyError(
            f"No pricing on file for {model!r}. "
            f"Known models: {known}. "
            f"Add it to PRICING in utils/pricing.py (check the pricing page first)."
        )
    price = PRICING[model]
    input_cost = input_tokens / 1_000_000 * price.input_per_1m
    output_cost = output_tokens / 1_000_000 * price.output_per_1m
    return input_cost + output_cost


def estimate_embedding_cost(model: str, input_tokens: int) -> float:
    """Return the estimated cost in USD for embedding `input_tokens` tokens.

    Embeddings have no output tokens, so cost depends only on the input.
    """
    if model not in EMBEDDING_PRICING:
        known = ", ".join(sorted(EMBEDDING_PRICING))
        raise KeyError(
            f"No embedding pricing on file for {model!r}. "
            f"Known models: {known}. "
            f"Add it to EMBEDDING_PRICING in utils/pricing.py (check the pricing page first)."
        )
    return input_tokens / 1_000_000 * EMBEDDING_PRICING[model]


def format_cost(usd: float) -> str:
    """Pretty-print a cost. Tiny amounts get more decimal places so they don't
    just show up as ``$0.00`` and look free (they aren't!)."""
    if usd < 0.01:
        return f"${usd:.6f}"
    return f"${usd:.4f}"


if __name__ == "__main__":
    # Run `python utils/pricing.py` for a quick demo / sanity check.
    demo_model = "gpt-4o-mini"
    cost = estimate_cost(demo_model, input_tokens=1_000, output_tokens=500)
    print(f"{demo_model}: 1,000 in + 500 out  ->  {format_cost(cost)}")

    embed_model = "text-embedding-3-small"
    embed_cost = estimate_embedding_cost(embed_model, input_tokens=1_000)
    print(f"{embed_model}: 1,000 in  ->  {format_cost(embed_cost)}")
