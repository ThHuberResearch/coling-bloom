import asyncio
import enum
import os
from json import JSONDecodeError
from typing import List, Optional

import aiohttp
from transformers import AutoTokenizer


async def POST_request(
        url: str,
        session: aiohttp.ClientSession,
        headers: dict,
        payload: dict,
):
    async with session.post(url, json=payload, headers=headers) as response:
        try:
            resp = await response.json(content_type=None)
        except JSONDecodeError:
            return {'response': 'ERROR - JSONDecodeError'}
        return resp


async def prepare_embedding_requests(
        prompt: dict[str, dict[str, str]]
):
    url = os.environ['EMBEDDING_ENDPOINT']
    connector = aiohttp.TCPConnector(limit=500)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        headers = {
            'Authorization': os.environ['EMBEDDING_BEARER']
        }
        task = asyncio.create_task(POST_request(url, session, headers, prompt))
        tasks.append(task)
        responses = await asyncio.gather(*tasks)
        return responses


class GenerationModel(enum.Enum):
    LLAMA = 'meta-llama/Llama-2-70b-chat-hf'
    LLAMA3 = 'meta-llama/Meta-Llama-3-70B-Instruct'
    MIXTRAL = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    GPT4 = 'gpt4'
    CLAUDE3 = 'claude3'
    BLOOMBERT = 'bloombert'
    GPT4o = 'gpt4o'


async def prepare_requests(
        prompts: List[str],
        max_tokens: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model: GenerationModel = GenerationModel.LLAMA
):
    if model == GenerationModel.LLAMA:
        url = os.environ['LLAMA_GENERATE_ENDPOINT']
    elif model == GenerationModel.MIXTRAL:
        url = os.environ['MIXTRAL_GENERATE_ENDPOINT']
    elif model == GenerationModel.LLAMA3:
        url = os.environ['LLAMA3_GENERATE_ENDPOINT']
    elif model == GenerationModel.GPT4:
        url = 'https://api.openai.com/v1/chat/completions'
    elif model == GenerationModel.GPT4o:
        url = 'https://api.openai.com/v1/chat/completions'
    elif model == GenerationModel.CLAUDE3:
        url = 'https://api.anthropic.com/v1/messages'
    elif model == GenerationModel.BLOOMBERT:
        url = 'https://bloom-bert-api-dmkyqqzsta-as.a.run.app/predict'
    else:
        raise NotImplementedError(f'Unknown model {model}')
    connector = aiohttp.TCPConnector(limit=500)
    timeout = aiohttp.ClientTimeout(total=60 * 60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for prompt in prompts:
            if tokenizer is not None:
                prompt_length = len(tokenizer(prompt)['input_ids'])
                if prompt_length > max_tokens:
                    continue  # skip for now
            headers = {
                'Authorization': 'Bearer ' + os.environ['LLAMA_API_KEY']
            }
            payload = {
                'prompt': prompt,
            }
            if max_tokens is not None:
                payload['maxTokens'] = max_tokens

            if model == GenerationModel.GPT4:
                headers = {
                    'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
                    'Content-Type': 'application/json',
                }
                payload = {
                    'model': 'gpt-4-0613',
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt,
                        },
                    ],
                }
                if max_tokens is not None:
                    payload['max_tokens'] = max_tokens
            elif model == GenerationModel.CLAUDE3:
                headers = {
                    'x-api-key': os.environ['CLAUDE_API_KEY'],
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json',
                }
                payload = {
                    # TODO: this is the SMALLEST model!
                    'model': 'claude-3-haiku-20240307',
                    'max_tokens': 1024,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt,
                        },
                    ],
                }
            elif model == GenerationModel.GPT4o:
                # gpt-4o-2024-05-13
                headers = {
                    'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
                    'Content-Type': 'application/json',
                }
                payload = {
                    'model': 'gpt-4o-2024-05-13',
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt,
                        },
                    ],
                }
                if max_tokens is not None:
                    payload['max_tokens'] = max_tokens
            elif model == GenerationModel.BLOOMBERT:
                headers = {}
                payload = {
                    'text': prompt,
                }
            task = asyncio.create_task(POST_request(url, session, headers, payload))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        return responses
