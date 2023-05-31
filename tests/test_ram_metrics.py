import math
import time

import torch
from avalanche.evaluation.metrics import Accuracy

acc = Accuracy()

acc.update([0], [1])
acc.update([1], [1])
acc.update([1], [1])

print(acc.result())
