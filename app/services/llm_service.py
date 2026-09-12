from loguru import logger
from openai import AsyncOpenAI, OpenAI

from app.config import settings

openai_client = OpenAI(api_key=settings.openai_api_key)
async_openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

def generate(system_prompt: str, user_message: str, model: str | None = None, temperature: float = 0.0) -> dict:
    if model is None:
        model = settings.llm_model_answer

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    logger.info(f"LLM response: {response.choices[0].message.content if response.choices else 'No content'}")

    text = response.choices[0].message.content or ""

    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    logger.info(f"Token usage: {usage}")

    return {"text": text, "usage": usage}

def generate_with_json(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:

    if model is None:
        model = settings.llm_model_grader

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return {"text": text, "usage": usage}

async def generate_async(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    if model is None:
        model = settings.llm_model_answer

    response = await async_openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    
    content = response.choices[0].message.content if response.choices else "No content"
    logger.info(f"LLM response: {content}")

    text = response.choices[0].message.content or ""

    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    logger.info(f"Token usage: {usage}")

    return {"text": text, "usage": usage}