import asyncio
import random
import time

from websockets import connect


class WebSocketClient:
    async def send(self, websocket):
        while True:
            message = f"im client {time.time()}"
            await websocket.send(message)
            print(f"Sent message: {message}")
            await asyncio.sleep(random.randint(0, 10))

    async def receive(self, websocket):
        while True:
            response = await websocket.recv()
            print(f"Received response: {response}")

    async def communicate(self):
        async with connect("ws://localhost:8001") as websocket:
            send_task = asyncio.create_task(self.send(websocket))
            receive_task = asyncio.create_task(self.receive(websocket))
            done, pending = await asyncio.wait(
                [send_task, receive_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()



if __name__ == "__main__":
    client = WebSocketClient()
    asyncio.run(client.communicate())
