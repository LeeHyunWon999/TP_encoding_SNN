import sys
import os
import json
import numpy as np

import torch
from spikingjelly.activation_based import neuron, encoding

# MIT-BIH train, test 분리

inputPath = "/data/leehyunwon/MIT-BIH_split/mitbih_train_data.npz"
outputPath = "/data/leehyunwon/MIT-BIH_split"

# 경로분리
fileName = os.path.basename(inputPath).split('.')[0:-1] # 파일명에 .이 들어간 경우를 위한 처리
fileName = ".".join(fileName)
# 데이터 파일 읽기 시도
inputData = np.load(inputPath)

# shape 확인
print(inputData['x'][0])
# print(len(inputData['y']))
print(inputData['x'])
