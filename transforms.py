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
        texts = texts[:self.max_length]
        texts = self.toTensor(texts)
        texts = self.padding(texts)
        return texts


class ImageTransform(nn.Module):
    def __init__(self):
        super(ImageTransform, self).__init__()
        self.resize = it.Resize(256)
        self.toTensor = it.ToTensor()
        self.normalize = it.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
