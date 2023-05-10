import torch
import torchtext.transforms as tt
import torchvision.transforms as it
import torch.nn as nn


class TextTransform(nn.Module):
    def __init__(self, max_length: int = 20, pad_values: int = 0):
        super(TextTransform, self).__init__()
        self.max_length = max_length
        self.toTensor = tt.ToTensor()
        self.padding = tt.PadTransform(max_length, pad_values)

    def forward(self, texts):
        texts = texts[: self.max_length]
        texts = self.toTensor(texts)
        texts = self.padding(texts)
        return texts


class ImageTransform(nn.Module):
    def __init__(self):
        super(ImageTransform, self).__init__()
        self.toTensor = it.ToTensor()
        self.resize = it.Resize((256, 256), antialias=True)

    def forward(self, images):
        images = self.toTensor(images)
        images = self.resize(images)
        return images / 255
