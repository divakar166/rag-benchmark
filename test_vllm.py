import os
import time
from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:8002/v1")
    api_key = os.getenv("LLM_API_KEY", "dummy")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")

    print("Testing vLLM endpoint")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print("-" * 60)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # 1. Check /models
    print("1. Checking /models endpoint...")

    try:
        models = client.models.list()
        available_models = [m.id for m in models.data]

        print("Available models:")
        for m in available_models:
            print(f"  - {m}")

        if model not in available_models:
            print()
            print(f"WARNING: Expected model not found: {model}")
            print("This may still work if your vLLM server aliases the model differently.")
        else:
            print("Model found.")
    except Exception as e:
        print("Failed to fetch models.")
        print(f"Error: {e}")
        return

    print("-" * 60)

    # 2. Chat completion test
    print("2. Testing chat completion...")

    start = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise assistant. Answer in one sentence.",
                },
                {
                    "role": "user",
                    "content": "What is RAG in simple words?",
                },
            ],
            max_tokens=128,
            temperature=0.2,
            top_p=0.7,
        )

        elapsed = time.time() - start

        answer = response.choices[0].message.content

        print("Response received.")
        print(f"Latency: {elapsed:.2f} seconds")
        print()
        print("Model response:")
        print(answer)

    except Exception as e:
        print("Chat completion failed.")
        print(f"Error: {e}")
        return

    print("-" * 60)

    # 3. Basic quality/format test
    print("3. Testing instruction following...")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return only valid JSON with these keys: "
                        "status, provider, working. "
                        "provider should be 'vllm'. working should be true."
                    ),
                }
            ],
            max_tokens=128,
            temperature=0,
        )

        text = response.choices[0].message.content

        print("Raw response:")
        print(text)

    except Exception as e:
        print("Instruction-following test failed.")
        print(f"Error: {e}")
        return

    print("-" * 60)
    print("vLLM endpoint test completed.")


if __name__ == "__main__":
    main()