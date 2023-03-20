import io
import json
import requests
from PIL import Image
import numpy as np
import torch

from torchvision import datasets, transforms
data_path = "C:\\Users\\inoue\\Documents\\AI_Learning_Dataset\\"
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)

cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label])
          for img, label in cifar10
          if label in [0, 2]]
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val
              if label in [0, 2]]

img, label = cifar2[2] # PIL_Image, Label
img.show()

print("cifar2[0] ", (img, label))

np_img = np.array(img).transpose(2,0,1) # (H,W,C) -> (C,H,W)
print("shape: ", list(np_img.shape))

meta = io.StringIO(json.dumps({'shape': list(np_img.shape)}))
buffer = io.BytesIO(bytearray(np_img))

r = requests.post("http://localhost:8000/predict", files={'meta': meta, 'data': buffer})

response = json.loads(r.content)

print("Model predicted probabilities of airplane vs bird:")
print(response)


