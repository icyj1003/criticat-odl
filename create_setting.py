import json

dropout_rate = 0.5
num_epochs = 50
text_arch = ["lstm", "bilstm", "gru", "bigru", "cnn"]
text_feature = ["train", "pho", "bert"]
image_arch = ["resnet18", "resnet34"]
model_type = ["text", "image", "metadata", "user_name"]
buffer_type = ["none", "full", "fixed", "classes"]

for _type in model_type:
    if _type == "text":
        for tarch in text_arch:
            for 