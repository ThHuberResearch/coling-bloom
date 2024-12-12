from typing import Optional

from util.template import PromptTemplate


class DirectPrompt(PromptTemplate):
    def __init__(
            self,
            model: str,
            temperature: float = 0.7,
            *args,
            **kwargs
    ):
        super().__init__(model, temperature=temperature, *args, **kwargs)

    def do_prompting(
            self,
            prompt: str,
            *args,
            **kwargs
    ) -> str:
        return self.send_prompt_to_model(prompt)
