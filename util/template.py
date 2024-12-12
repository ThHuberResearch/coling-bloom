import abc
import asyncio
import os
from typing import Optional

import anthropic
import requests
from langchain_openai import ChatOpenAI
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from util.api_stuff import prepare_requests, GenerationModel


class PromptTemplate(abc.ABC):
    def __init__(
            self,
            model: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            *args,
            **kwargs
    ):
        if model not in [
            'llama',
            'llama3',
            'gpt-4',
            'gpt-4o-2024-05-13',
            'mixtral',
            'gemma',
            'claude3',
            'gpt-3.5-turbo-0125',
            'davinci-002'
        ]:
            try:
                model_instance = AutoModelForSeq2SeqLM.from_pretrained(model)
            except OSError:
                raise NotImplementedError(f'Unknown HuggingFace model: {model}')
            except ValueError:
                print("> Model can't be loaded as AutoModelForSeq2SeqLM, trying AutoModelForCausalLM")
                try:
                    model_instance = AutoModelForCausalLM.from_pretrained(model)
                except ValueError:
                    raise NotImplementedError(
                        f"HuggingFace model can't be loaded as AutoModelForSeq2SeqLm OR AutoModelForCausalLM: {model}"
                    )
            self.is_huggingface = True
            self.model_instance = model_instance
            self.tokenizer_instance = AutoTokenizer.from_pretrained(model)
        else:
            self.is_huggingface = False
        if model in [
            'gpt-4',
            'gpt-3.5-turbo-0125',
            'gpt-4o-2024-05-13'
        ]:
            self.openai_client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY']
            )
        else:
            self.openai_client = None
        if model == 'claude3':
            self.claude_client = anthropic.Anthropic(
                api_key=os.environ['CLAUDE_API_KEY'],
            )
        else:
            self.claude_client = None
        if model == 'gemma':
            if not os.environ.get('DISABLE_GEMMA_WARNING', False):
                print(
                    f'> WARNING: Gemma must be running! It is not always on, so make sure it is running before using it.')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.args = args
        self.kwargs = kwargs
        # print(f'Initialized {self.__class__.__name__} with model {model}')

    def format_gemma(
            self,
            prompt: str,
            demonstrations: Optional[list[str]] = None,
            system_prompt: Optional[str] = None
    ):
        template = '''<start_of_turn>user
{prompt}
'''
        if system_prompt is not None:
            prompt += '\n' + system_prompt
        template = template.replace('{prompt}', prompt)
        if demonstrations:
            template += '\n[Examples]' + '\n'.join(demonstrations) + '\n[/Examples]\n'
        template += '<end_of_turn>\n<start_of_turn>model'
        return template

    def format_prompt(
            self,
            prompt: str,
            demonstrations: Optional[list[str]] = None,
            system_prompt: Optional[str] = None
    ) -> str:
        if self.model == 'gemma':
            return self.format_gemma(prompt, demonstrations, system_prompt)
        else:
            return prompt

    @abc.abstractmethod
    def do_prompting(
            self,
            prompt: str,
            demonstrations: Optional[list[str]] = None,
            *args,
            **kwargs
    ) -> str:
        pass

    def send_prompt_to_model(
            self,
            prompt: str
    ) -> str:
        if self.model == 'llama':
            return self._prompt_llama(prompt)
        elif self.model == 'llama3':
            return self._prompt_llama3(prompt)
        elif self.model == 'gpt-4':
            return self._prompt_gpt4(prompt)
        elif self.model == 'gpt-3.5-turbo-0125':
            return self._prompt_gpt35(prompt)
        elif self.model == 'gpt-4o-2024-05-13':
            return self._prompt_gpt4o(prompt)
        elif self.model == 'mixtral':
            return self._prompt_mixtral(prompt)
        elif self.model == 'gemma':
            return self._prompt_gemma(prompt)
        elif self.model == 'claude3':
            return self._prompt_claude3(prompt)
        elif self.model == 'davinci-002':
            return self._prompt_davinci002(prompt)
        elif self.is_huggingface:
            return self._prompt_huggingface(prompt)
        else:
            raise NotImplementedError(f'Unknown model {self.model}')

    def _prompt_llama(
            self,
            prompt: str
    ) -> str:
        llm = ChatOpenAI(
            model_name='meta-llama/Llama-2-70b-chat-hf',
            openai_api_base=os.environ['LLAMA_GENERATE_ENDPOINT'],
            openai_api_key=os.environ['LLAMA_API_KEY'],
            temperature=self.temperature
        )
        out = llm.invoke(prompt)
        return out.content

    def _prompt_llama3(
            self,
            prompt: str
    ) -> str:
        if self.max_tokens is None:
            max_tokens = 2000
        else:
            max_tokens = self.max_tokens
        out = asyncio.run(
            prepare_requests(
                [prompt],
                model=GenerationModel.LLAMA3,
                max_tokens=max_tokens
            )
        )
        return out[0]['response']

    def _prompt_gpt4(
            self,
            prompt: str
    ) -> str:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        return completion.choices[0].message.content

    def _prompt_gpt4o(
            self,
            prompt: str
    ) -> str:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        return completion.choices[0].message.content

    def _prompt_gpt35(
            self,
            prompt: str
    ) -> str:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        return completion.choices[0].message.content

    def _prompt_davinci002(
            self,
            prompt: str
    ) -> str:
        api_key = os.environ['OPENAI_API_KEY']
        url = 'https://api.openai.com/v1/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
        data = {
            'prompt': prompt,
            'max_tokens': 500,
            'model': 'davinci-002'
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['text']

    # noinspection PyMethodMayBeStatic
    def _prompt_mixtral(
            self,
            prompt: str
    ) -> str:
        out = asyncio.run(
            prepare_requests(
                [prompt],
                model=GenerationModel.MIXTRAL,
                max_tokens=2000
            )
        )
        return out[0]['response']

    def _prompt_gemma(
            self,
            prompt: str
    ) -> str:
        out = asyncio.run(
            prepare_requests(
                [prompt],
                model=GenerationModel.MIXTRAL,  # this is INTENTIONAL - uses same endpoint as mixtral
                max_tokens=2000
            )
        )
        return out[0]['response']

    def _prompt_huggingface(
            self,
            prompt: str
    ) -> str:
        max_new_tokens = self.kwargs.get('max_new_tokens', 2000)
        inputs = self.tokenizer_instance(prompt, return_tensors='pt')
        outputs = self.model_instance.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer_instance.batch_decode(outputs, skip_special_tokens=True)[0]

    def _prompt_claude3(
            self,
            prompt: str
    ) -> str:
        message = self.claude_client.messages.create(
            model='claude-3-opus-20240229',
            max_tokens=self.kwargs.get('max_new_tokens', 1000),
            temperature=self.temperature,
            system='',  # intentionally blank
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        message_text = message.content[0].text
        return message_text
