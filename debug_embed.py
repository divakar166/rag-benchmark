import asyncio
from config import settings
from langchain_openai import OpenAIEmbeddings

async def main():
    print("host:", settings.embedding_host)
    print("model:", settings.embedding_model)

    emb = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key="not-needed",
        openai_api_base="http://localhost:8001/v1",
        check_embedding_ctx_length=False,
    )

    vec = await emb.aembed_query("test")
    print(len(vec), vec[:5])

asyncio.run(main())