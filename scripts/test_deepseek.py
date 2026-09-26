#!/usr/bin/env python3
"""
Test connection to DeepSeek API (OpenAI-compatible).
Usage:
    python scripts/test_deepseek.py
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from core.config import (
    get_red_provider,
    get_red_model,
    get_openai_api_key,
    red_openai_client_kwargs,
)

def test_deepseek_connection():
    api_key = get_openai_api_key()
    provider = get_red_provider()
    model = get_red_model()
    kwargs = red_openai_client_kwargs()

    print("=" * 60)
    print("Testing DeepSeek / LLM API Connection")
    print("=" * 60)
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print(f"Base URL : {kwargs.get('base_url', 'default (api.openai.com)')}")
    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "(not set or invalid)"
    print(f"API Key  : {masked_key}")

    if not api_key or api_key.startswith("sk-...") or api_key == "sk-your-key":
        print("\n[!] LƯU Ý: Bạn chưa điền API Key thật vào file .env!")
        print("    Vui lòng mở file .env và điền DEEPSEEK_API_KEY hoặc OPENAI_API_KEY.")
        return False

    try:
        from openai import OpenAI
        client = OpenAI(**kwargs)
        print("\nĐang gửi request kiểm tra tới DeepSeek...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Please reply in 1 short sentence confirming you are DeepSeek."},
            ],
            max_tokens=60,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        print("\n[SUCCESS] Kết nối DeepSeek thành công!")
        print(f"Phản hồi từ model: {content}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Kết nối thất bại: {e}")
        return False

if __name__ == "__main__":
    test_deepseek_connection()
