from kafka import KafkaProducer
import json
import time 

TOPIC = "ABC"

producer = KafkaProducer(
    bootstrap_servers=["localhost:9093"],
    key_serializer=lambda k: json.dumps(k).encode("utf-8"),
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

while True:
    producer.send(topic=TOPIC, key="XYZ", value={"text": "hello"})
    time.sleep(1)