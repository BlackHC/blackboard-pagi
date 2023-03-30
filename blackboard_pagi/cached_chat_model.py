from typing import List, Optional

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, ChatGeneration, ChatResult, Generation


class CachedChatOpenAI(ChatOpenAI):
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        messages_prompt = repr(messages)
        if langchain.llm_cache:
            results = langchain.llm_cache.lookup(messages_prompt, self.model_name)
            if results:
                assert len(results) == 1
                result: Generation = results[0]
                chat_result = ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=result.text))],
                    llm_output=result.generation_info,
                )
                return chat_result
        chat_result = super()._generate(messages, stop)
        if langchain.llm_cache:
            assert len(chat_result.generations) == 1
            result = Generation(text=chat_result.generations[0].message.content, generation_info=chat_result.llm_output)
            langchain.llm_cache.update(messages_prompt, self.model_name, [result])
        return chat_result
