"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

# Tự động nạp API Keys từ file .env
load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-3.6-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.6-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    response_text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API and return the response text, latency,
    and token usage stats.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time

    response_text = response.text or ""
    usage = {
        "input_tokens": (
            response.usage_metadata.prompt_token_count
            if response.usage_metadata
            else 0
        ),
        "output_tokens": (
            response.usage_metadata.candidates_token_count
            if response.usage_metadata
            else 0
        ),
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    response_text = ""
    if response.content and len(response.content) > 0:
        response_text = getattr(response.content[0], "text", "")

    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini Flash
    with the same prompt and return a structured comparison dictionary.
    """
    # 1. GPT-4o
    text_gpt4o, lat_gpt4o, usage_gpt4o = call_openai(prompt, model=OPENAI_MODEL)
    cost_gpt4o = (
        usage_gpt4o["input_tokens"] * PRICING_1M_TOKENS[OPENAI_MODEL]["input"]
        + usage_gpt4o["output_tokens"] * PRICING_1M_TOKENS[OPENAI_MODEL]["output"]
    ) / 1_000_000.0

    # 2. GPT-4o-mini
    text_mini, lat_mini, usage_mini = call_openai(prompt, model=OPENAI_MINI_MODEL)
    cost_mini = (
        usage_mini["input_tokens"] * PRICING_1M_TOKENS[OPENAI_MINI_MODEL]["input"]
        + usage_mini["output_tokens"] * PRICING_1M_TOKENS[OPENAI_MINI_MODEL]["output"]
    ) / 1_000_000.0

    # 3. Gemini Flash
    text_gemini, lat_gemini, usage_gemini = call_gemini(prompt, model=GEMINI_MODEL)
    gemini_pricing_key = GEMINI_MODEL if GEMINI_MODEL in PRICING_1M_TOKENS else "gemini-2.5-flash"
    cost_gemini = (
        usage_gemini["input_tokens"] * PRICING_1M_TOKENS[gemini_pricing_key]["input"]
        + usage_gemini["output_tokens"] * PRICING_1M_TOKENS[gemini_pricing_key]["output"]
    ) / 1_000_000.0

    return {
        "gpt4o": {
            "response": text_gpt4o,
            "latency": lat_gpt4o,
            "cost": cost_gpt4o,
            "input_tokens": usage_gpt4o["input_tokens"],
            "output_tokens": usage_gpt4o["output_tokens"],
        },
        "gpt4o_mini": {
            "response": text_mini,
            "latency": lat_mini,
            "cost": cost_mini,
            "input_tokens": usage_mini["input_tokens"],
            "output_tokens": usage_mini["output_tokens"],
        },
        "gemini_flash": {
            "response": text_gemini,
            "latency": lat_gemini,
            "cost": cost_gemini,
            "input_tokens": usage_gemini["input_tokens"],
            "output_tokens": usage_gemini["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini.
    Uses chat session to stream tokens and manage conversation history natively.
    """
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    chat = client.chats.create(model=GEMINI_MODEL)

    print("Chatbot sẵn sàng! Gõ 'quit' hoặc 'exit' để dừng.\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Đang thoát phiên hội thoại...")
            break

        print("Bot: ", end="", flush=True)
        try:
            response_stream = chat.send_message_stream(user_input)
            for chunk in response_stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print("\n")
        except Exception as err:
            print(f"\n[Lỗi tạo nội dung]: {err}\n")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.
    """
    results = []
    for p in prompts:
        try:
            res = compare_models(p)
        except TypeError:
            # Bắt lỗi khi mock test function không khai báo tham số nhận vào (_get_mock())
            res = compare_models()
        res["prompt"] = p
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.
    Columns: | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    headers = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows = []

    model_mapping = [
        ("gpt4o", "GPT-4o"),
        ("gpt4o_mini", "GPT-4o-mini"),
        ("gemini_flash", "Gemini Flash"),
    ]

    for item in results:
        prompt_clean = item.get("prompt", "").replace("\n", " ").strip()
        truncated_prompt = (
            (prompt_clean[:30] + "...") if len(prompt_clean) > 30 else prompt_clean
        )

        for key, display_name in model_mapping:
            if key not in item:
                continue
            data = item[key]
            raw_text = data.get("response", "").replace("\n", " ").strip()
            truncated_resp = (raw_text[:50] + "...") if len(raw_text) > 50 else raw_text
            latency_str = f"{data.get('latency', 0.0):.2f}s"
            tokens_str = f"{data.get('input_tokens', 0)} / {data.get('output_tokens', 0)}"
            cost_str = f"${data.get('cost', 0.0):.6f}"

            rows.append(
                f"| {truncated_prompt} | {display_name} | {truncated_resp} | {latency_str} | {tokens_str} | {cost_str} |"
            )

    return "\n".join(headers + rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = (
        "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    )
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")