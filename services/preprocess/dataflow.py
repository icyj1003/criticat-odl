import json
from bytewax.inputs import KafkaInputConfig
from bytewax.outputs import StdOutputConfig
from bytewax.dataflow import Dataflow
from bytewax.execution import run_main

# from preprocess import Cleaner
import os


# * Functions
def mapping(message):
    key, value = message
    k, v = json.loads(key), json.loads(value)
    # return cleaner.clean_one(value["content"])
    return value


# * Init cleaner
# cur_dir = os.path.abspath(os.curdir)
# cleaner = Cleaner(
#     stopwords_path="./vietnamese-stopwords-dash.txt",
#     vncorenlp_path="./vncorenlp/",
#     cur_dir=cur_dir,
# )

# * Dataflow configuration
flow = Dataflow()
flow.input(
    "input",
    KafkaInputConfig(topic="topic", brokers=["localhost:9093"], starting_offset="end"),
)
flow.map(mapping)
flow.capture(StdOutputConfig())

if __name__ == "__main__":
    run_main(flow)
