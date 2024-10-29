import sys
import os
import json
import numpy as np

import torch
from spikingjelly.activation_based import neuron, encoding

# 입력모듈용으로 리팩토링 필요 : 얘만의 json이 필요한 것이 아니기 때문이며 텐서로 바꾸는 등의 작업이 따로 필요하다.


# json 읽어다가 반환(파일경로 없으면 에러띄우기)
def loadJson() : 
    if (len(sys.argv) != 2) : 
        print("config.json 파일 경로가 없거나 그 이상의 인자가 들어갔습니다!", len(sys.argv))
        exit()
    else : 
        with open(sys.argv[1], 'r') as f:
            print("config.json파일 읽기 성공!")
            return json.load(f)
        

# 인코딩하고 결과 저장까지 진행
def encode(json_data) : 
    # 파일명 분리
    fileName = os.path.basename(json_data["inputPath"]).split('.')[0:-1] # 파일명에 .이 들어간 경우를 위한 처리
    fileName = ".".join(fileName)
    # 데이터 파일 읽기 시도
    inputData = np.loadtxt(json_data["inputPath"], delimiter=',')
    
    # 188째 값은 없앤다.
    inputData = np.delete(inputData, 187, axis=1)
    print(inputData.shape)
    
    # 파일 형변환
    inputData = torch.tensor(inputData)

    # 확률 변형 시도 : 50%, 25%
    inputData = inputData * 0.25

    
    encoded_list = []
    
    # 포아송은 그냥 이미 있는 인코더 쓰면 된다.
    encoder = encoding.PoissonEncoder()
    
    # 포아송은 확률적으로 출력하므로 이걸 반복해야 한다.
    for i in range(json_data['dim']) : 
        encoded_data = encoder(inputData)
    
        print(inputData.shape)
        print(encoded_data)
        encoded_list.append(encoded_data.numpy())
        print(len(encoded_list))
        print(len(encoded_list[0]))
        print(len(encoded_list[0][0]))
    


    print("인코딩 완료")
    
    
    
    
    # npy 형태로 통일
    encoded_array = np.array(encoded_list)
    # 출력데이터 또한 그 순서를 좀 바꾸도록 하자. 이 포아송 녀석은 (T, 데이터갯수, 뉴런) 인데, 이걸 (데이터갯수, 뉴런, T) 이걸로 바꿔야겠다.
    encoded_array = encoded_array.transpose(1, 2, 0)
    
    # 잘 되는지 출력필요
    print(encoded_array)
    print(encoded_array.shape)
    
    # npy로 저장
    np.save(json_data["outputPath"] + fileName + '_' + str(json_data["dim"]) + '_timestep_Poisson_prob25.npy', encoded_array) # 일단 이거 되긴 하는지 확인 필요



    print("저장 완료")



if __name__ == "__main__" : 
    json_data = loadJson()
    # config 파일 출력
    print(json_data)

    encode(json_data) # 인코딩 및 저장 함수