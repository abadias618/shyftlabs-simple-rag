from langchain_core.callbacks import AsyncCallbackHandler
import asyncio

class TokenStreamHandler(AsyncCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()
        
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        await self.queue.put(token)

    async def on_llm_end(self, response, **kwargs) -> None:
        await self.queue.put(None)