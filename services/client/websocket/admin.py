import asyncio
import json

import websockets


async def handler(websocket):
    print("New connection to websocket")
    try:
        async for message in websocket:
            data = json.loads(message)
            print(data)
    except websockets.exceptions.ConnectionClosedError:
        print("Connection closed")


async def main():
    async with websockets.serve(handler, "", 8001):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
