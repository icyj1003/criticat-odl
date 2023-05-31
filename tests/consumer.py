from kafka import KafkaConsumer

topic = "topic"

consumer = KafkaConsumer(
    topic, bootstrap_servers=["localhost:9093"], auto_offset_reset="latest"
)

for message in consumer:
    print(message)
