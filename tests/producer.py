import time

from kafka import KafkaProducer

topic = "topic"

producer = KafkaProducer(bootstrap_servers=["localhost:9093"])
for _ in range(100):
    producer.send(topic, b"some_message_bytes")
    time.sleep(1)
