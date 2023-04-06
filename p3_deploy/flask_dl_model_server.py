"""サーバークライアントモデルのDLサーバー
"""

import os
import sys
import json
from typing import *
from typing_extensions import *
from flask import Flask, jsonify, request
import numpy as np
import torch
from torchvision import transforms

# ネットワークのモデル定義
from flask_deploy_model import ConvNet

# データ成形
transform=transforms.Compose([
        # transforms.ToTensor(), # IN [PILImage, numpy] -> OUT scaled Tensor[0,1]
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ])

app = Flask(__name__)

model = ConvNet()
model.load_state_dict(
    torch.load(
    "C:\\Users\\inoue\\Documents\\MyGithub\\Book_HowToPytorch\\weights\\p1ch8\\bird_vs_airplanes.pt",
    map_location='cpu'))

model.eval()

# 推論
def run_inference(in_tensor: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        out_tensor = model(in_tensor.unsqueeze(0)).squeeze(0) # (N,2) nn.LogSoftmax
        probs = torch.exp(out_tensor).tolist() # probabirity
    out = dict()
    out['airplane'] = probs[0]
    out['bird'] = probs[1]
    return out
    
# POSTで公開
@app.route("/predict", methods=["POST"])
def predict():
    # HTTPリクエストを取得
    meta = json.load(request.files['meta'])
    # データ読み込み
    bin_img = request.files['data'].read() # binary
    # Tensor化
    in_tensor = torch.from_numpy(np.frombuffer(bin_img, dtype=np.uint8)) # RGB
    print("Server: in_tensor type", type(in_tensor))
    print("Server: in_tensor shape", in_tensor.size())

    # Tensorの形状変更&正規化
    in_tensor = in_tensor.view(*meta['shape'])
    in_tensor = in_tensor.to(torch.float32) # uint8 -> float32
    in_tensor = (in_tensor - in_tensor.min()) / (in_tensor.max() - in_tensor.min()) # [0,255] -> [0,1]
    # print("in_tensor:\n", in_tensor)
    in_tensor = transform(in_tensor) # 標準化

    # 推論
    out = run_inference(in_tensor)
    print("Inferenced out: ", out)

    # HTTPレスポンス
    return jsonify(out)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)