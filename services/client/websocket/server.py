import asyncio
import json

import websockets
from pymongo import MongoClient

# SETTINGS
MONGO_CONNECT_STRING = "mongodb://localhost:27017/"
DEFAULT_PORT = 8001

class Server:
    clients = set()
    mongo = MongoClient(MONGO_CONNECT_STRING)

    def __init__(self) -> None:
        self.db = self.mongo["CritiCat"]
        self.raw_coll = self.db["raw"]

    async def send(self, websocket):
        try:
            # TODO: replace with the Kafka consumer: list listen to label -> 
            while True:
                # send dummy message
                await websocket.send("hello")

                # delay
                await asyncio.sleep(2)

        except websockets.exceptions.ConnectionClosedOK:
            print("Client disconnected")

    async def receive(self, websocket):
        try:
            # TODO: listen to 2 different type of messages: raw text and feedback signals
            while True:
                # read raw response from client
                raw_response = await websocket.recv()

                # convert raw response to json object
                article = json.loads(raw_response)

                # insert new article to database
                self.raw_coll.insert_one(article)

                # send signal to bigcat

        except websockets.exceptions.ConnectionClosedOK:
            print("Client disconnected")


    async def handler(self, websocket):
        # * add new client to clients list 
        self.clients.add(websocket)

        # * create parallel tasks for client
        send_task = asyncio.create_task(self.send(websocket))
        receive_task = asyncio.create_task(self.receive(websocket))

        # * terminate both tasks when any of them have completed
        done, pending = await asyncio.wait(
            [send_task, receive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # * cancel them
        for task in pending:
            task.cancel()

    async def communicate(self):
        async with websockets.serve(self.handler, "", DEFAULT_PORT):
            await asyncio.Future()


if __name__ == "__main__":
    server = Server()
    asyncio.run(server.communicate())
