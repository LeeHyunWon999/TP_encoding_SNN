import sys
import os
import json
import numpy as np

import torch
from spikingjelly.activation_based import neuron, encoding

# MIT-BIH train, test 분리

inputPath = "/data/common/MIT-BIH/mitbih_train.csv"
outputPath = "/data/leehyunwon/MIT-BIH_split"

# 경로분리
fileName = os.path.basename(inputPath).split('.')[0:-1] # 파일명에 .이 들어간 경우를 위한 처리
fileName = ".".join(fileName)
# 데이터 파일 읽기 시도
inputData = np.loadtxt(inputPath, delimiter=',')

# shape 확인
# print(inputData.shape)  -> 결과 : 21892, 188

# 인덱스 1(188)의 마지막 열만 따로 떼서 y로, 나머진 x로 저장
x = inputData[:, :-1]  # 모든 행과 마지막 열을 제외한 열들
y = inputData[:, -1]   # 모든 행과 마지막 열만


# 이진값으로 변경
for i in range(len(y)) : 
    if y[i] == 0 : 
        y[i] = 0
    else : 
        y[i] = 1



# 각각의 데이터를 .npz 파일로 저장
np.savez(os.path.join(outputPath, f"{fileName}_data.npz"), x=x)
np.savez(os.path.join(outputPath, f"{fileName}_labels.npz"), y=y)

print(f"Data saved to {outputPath}/{fileName}_data.npz")
print(f"Labels saved to {outputPath}/{fileName}_labels.npz")