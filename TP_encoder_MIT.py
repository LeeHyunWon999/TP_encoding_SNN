import sys
import os
import json
import numpy as np

import torch
from spikingjelly.activation_based import neuron

# 똑같은 TP 인코더, 근데 이제 2차원용으로 차원 확장된


# 인코딩용 뉴런 정의
class TP_neuron(neuron.BaseNode) : 
    
    # 생성자
    def __init__(self, tau, g, threshold = 1.0, reset = False, reset_value = 0.0, leaky = False) : 
        super().__init__()
        self.v_threshold = threshold
        self.v_reset = None if reset is False else reset_value
        self.leaky = leaky
        self.tau = tau
        self.g = g
    
    # 인코딩 스텝
    def neuronal_charge(self, x: torch.Tensor) : 
        if self.leaky : 
            # LIF 뉴런 체계인 경우 leak 값인 tau를 이용하여 계산
            self.v = np.exp(-1 / self.tau) * self.v + (x * self.g)
        else : 
            self.v = self.v + (x * self.g)
    


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
    
    # 파일 형변환
    inputData = torch.tensor(inputData)
    
    # 입력데이터는 [N, T] 형식인데, 인코더 뉴런은 [T, N, *] 을 기대한다. 따라서 축을 변경한다.
    inputData = inputData.transpose(1,0)
    
    print(inputData)

    # print(inputData)
    # print(fileName)
    
    # 임시 : 뉴런 생성(뉴런 여러개 만들기)
    # 임시 : 필요한 경우 내부 막전위값 변화를 timestep별로 보도록 할 수도 있겠지만.. 일단은 패스
    neuron_list = []
    encoded_list = []
    
    # 시간축을 이렇게 조지는게 내가 하는거라서.. ** 일단 이거부터 수정필요 **
    for i in range(json_data["dim"]) : 
        # Leaky인 경우 tau값은 0.5~1.5 사이에서 랜덤 지정
        if json_data["leaky"] : 
            this_tau = np.random.rand() + 0.5
        else : 
            this_tau = 1
        neuron_list.append(TP_neuron(tau = this_tau, g = ((float(i) + 1) / float(json_data["dim"])) + 0.5))
        neuron_list[i].step_mode = 'm'
        encoded_list.append(neuron_list[i](inputData).numpy())
        print(i,"째 뉴런(g=" + str(neuron_list[i].g) + ") 인코딩 결과 : ", encoded_list[i])
        neuron_list[i].reset()
    
    
    # 결과값(텐서) 의 첫 열과 둘째 열에 각각 tau, g 값 추가 : 일단 MIT-BIH 인코딩용이니 번거롭게 하지 말고 걍 제거, g값 자체도 코드로 유추 가능하니 더더욱.
    # for i in range(len(encoded_list)) : 
    #     encoded_list[i] = encoded_list[i].tolist()
    #     encoded_list[i].insert(0, neuron_list[i].g)
    #     encoded_list[i].insert(0, neuron_list[i].tau)
        
    print("인코딩 완료")
    
    
    
    
    # 임시 : csv로 저장(각 뉴런들의 결과 값 리스트 합치고 저장)
    # np.savetxt('./data/output/' + fileName + '_encoded.csv', encoded_list, fmt="%f", delimiter=',') -> 이건 csv로 저장하면 안되는 것 같다. 내껀 3차원이니깐..
    
    # npy 형태로 통일
    encoded_array = np.array(encoded_list)
    # 출력데이터 또한 그 순서를 좀 바꾸도록 하자. 지금은 (뉴런, T, 데이터갯수) 인데, 이걸 (데이터갯수, 뉴런, T) 이걸로 바꿔야겠다.
    encoded_array = encoded_array.transpose(2, 0, 1)
    
    # 잘 되는지 출력필요
    print(encoded_array)
    
    # npy로 저장
    np.save(json_data["outputPath"] + fileName + '_' + str(json_data["dim"]) + '_encoded.npy', encoded_array) # 일단 이거 되긴 하는지 확인 필요



    print("저장 완료")



if __name__ == "__main__" : 
    json_data = loadJson()
    # config 파일 출력
    print(json_data)

    encode(json_data) # 인코딩 및 저장 함수