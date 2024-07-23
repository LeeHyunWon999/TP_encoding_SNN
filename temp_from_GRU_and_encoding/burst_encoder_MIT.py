import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt # burst 인코딩 잘 되나 확인용

import torch
from torch import Tensor, nn
from spikingjelly.activation_based import neuron

# 똑같은 TP 인코더, 근데 이제 2차원용으로 차원 확장된 기본버전

# 입력모듈용으로 리팩토링 필요 : 얘만의 json이 필요한 것이 아니기 때문이며 텐서로 바꾸는 등의 작업이 따로 필요하다.


# 인코딩용 뉴런 정의
class BURST(nn.Module):
    def __init__(self, beta=2, init_th=0.0625, device='cuda') -> None:
        super().__init__()
        """Burst coding at the time step

        Args:
            data (torch.Tensor): the data transfomed into range `[0, 1]`
            mem (torch.Tensor): data transfomed into range `[0, 1]`
            t (int): time step
            beta (float, optional): . Defaults to 2.0.
            th (float, optional): _description_. Defaults to 0.125.
        """
        
        
        
        
        self.beta = beta
        self.init_th = init_th
        self.device = device
        
        # self.th = torch.tensor([]).to(self.device)
        # self.mem = torch.zeros(data_num_steps).to(self.device) # membrane potential initialization
        
    def burst_encode(self, data, t):
        if t==0:
            self.mem = data.clone().detach().to(self.device) # 이건 그대로
            self.th = torch.ones(self.mem.shape, device=self.device) * self.init_th # 밖에 있는 코드 가져오느라 이렇게 된듯
            
        self.output = torch.zeros(self.mem.shape).to(self.device) # 0, 1 단위로 보내기 위해 이게 필요(아래 코드에 쓰는 용도)
        
        fire = (self.mem >= self.th) # 발화여부 확인
        self.output = torch.where(fire, torch.ones(self.output.shape, device=self.device), self.output) # 발화됐으면 1, 아니면 0 놓는 녀석
        out = torch.where(fire, self.th, torch.zeros(self.mem.shape, device=self.device)) # 얜 이제 잔차로 리셋하는 원래 동작 위해서 있는 녀석
        self.mem -= out
        
        self.th = torch.where(fire, self.th * self.beta, torch.ones(self.th.shape, device=self.device)*self.init_th) # 연속발화시 2배로 늘리기, 아니면 다시 초기치로 이동

        # 입력값 재설정하고 싶으면 쓰기 : 원본에서도 이건 그냥 있었으니 냅둘 것
        if self.output.max() == 0:
            self.mem = data.clone().detach().to(self.device)
        
        # 반환 : 스파이크 뜬 그 출력용 녀석 내보내기
        return self.output.clone().detach()
    
    def forward(self, input:Tensor, t:int) -> Tensor:
        return self.burst_encode(input, t)
    


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
    
    # 현재 입력데이터는 [N, T] 형식이다.
    
    print(inputData.shape)
    
    
    ## 대충 burst 인코딩 하는 구간
    burst_encoder = BURST()
    encoded_list = []
    
    # 아마도 타임스텝을 밖에서 지정하면서 넣어야 하는 것으로 보인다.
    for i in range(json_data['dim']) : 
        encoded_data = burst_encoder(inputData, i).cpu()
        print(encoded_data.shape)
        encoded_list.append(encoded_data.numpy())
    
    


    print("인코딩 완료")
    print(len(encoded_list))
    print(len(encoded_list[0]))
    print(len(encoded_list[0][0]))
    # 즉, 여기서 데이터는 [T, 데이터갯수, 뉴런갯수] 의 형식으로 저장되어 있다.
    
    
    
    
    # npy 형태로 통일
    encoded_array = np.array(encoded_list)
    # 출력데이터 또한 그 순서를 좀 바꾸도록 하자. 지금은 (T, 데이터갯수, 뉴런) 인데, 이걸 (데이터갯수, 뉴런, T) 이걸로 바꿔야겠다.
    encoded_array = encoded_array.transpose(1, 2, 0)
    
    # 잘 되는지 출력필요
    print(encoded_array)
    
    # burst 이녀석은 안만진지 꽤 오래되었으니, 시각화를 간단히라도 해봐야 하지 않나 싶다.
    # 대상은 0째 녀석으로. 축도 맞춰놨으니 슬라이스 하나 뽑는 것은 쉬울 것이다.
    # 2차원 배열 시각화
    show_array = encoded_array[0]
    show_array = show_array.transpose(1,0)
    plt.imshow(show_array, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('64 x 187 Binary Array Visualization')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    # 그래프를 파일로 저장
    plt.savefig('burst_encoded_visualization.png')
    
    # 걍 화면에 표시해버리기
    temp_array = encoded_array[0]
    for row in range(len(temp_array)) : 
        for column in range(len(temp_array[0])) : 
            print(temp_array[row][column], end=' ')
        print()
    
    
    
    
    
    
    
    # 마지막으로 shape 확인
    print(encoded_array.shape)
    
    # npy로 저장
    np.save(json_data["outputPath"] + fileName + '_' + str(json_data["dim"]) + '_timestep_burst.npy', encoded_array) # 일단 이거 되긴 하는지 확인 필요



    print("저장 완료")



if __name__ == "__main__" : 
    
    # Cuda 써야겠지?
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # GPU 번호별로 0번부터 나열
    os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # 일단 원석이가 0, 1번 쓰고 있다 하니 2번으로 지정
    device = "cuda" if torch.cuda.is_available() else "cpu" # 연산에 GPU 쓰도록 지정
    print("Device :" + device) # 확인용
    
    json_data = loadJson()
    # config 파일 출력
    print(json_data)

    encode(json_data) # 인코딩 및 저장 함수