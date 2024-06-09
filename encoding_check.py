# 여기서 바로 쓰던 Linear 인코더에 값을 고정시켜서 테스트 순전파 했을 때 의도대로 잘 인코딩되는지 확인 위한 파일
# 따라서 학습(역전파), 텐서보드 찍기 등은 다 쳐내도 될 것이다.

# Imports
import os
import torch
import numpy as np # .npy 읽기용
import pandas as pd # csv 읽기용
import torch.nn.functional as F  # 일부 활성화 함수 등 파라미터 없는 함수에 사용
import torchvision.datasets as datasets  # 일반적인 데이터셋; 이거 아마 MIT-BIH로 바꿔야 할 듯?
import torchvision.transforms as transforms  # 데이터 증강을 위한 일종의 변형작업이라 함
from torch import optim  # SGD, Adam 등의 옵티마이저(그래디언트는 이쪽으로 가면 됩니다)
from torch.optim.lr_scheduler import CosineAnnealingLR # 코사인스케줄러(옵티마이저 보조용)
from torch import nn  # 모든 DNN 모델들
from torch.utils.data import (DataLoader, Dataset)  # 미니배치 등의 데이터셋 관리를 도와주는 녀석
from tqdm import tqdm  # 진행도 표시용
import torchmetrics # 평가지표 로깅용
from typing import Callable # 람다식
# from torch.utils.tensorboard import SummaryWriter # tensorboard 기록용

# 여긴 인코더 넣을때 혹시 몰라서 집어넣었음
import sys
import os
import json
import numpy as np

# 얘는 SNN 학습이니까 당연히 있어야겠지? 특히 SNN 모델을 따로 만드려는 경우엔 뉴런 말고도 넣을 것이 많다.
# import spikingjelly.activation_based as jelly
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

# 이쪽에선 SNN 모델을 넣지 않고, 바로 jelly.layer.Linear로 바로 들어가는 것을 시도해본다. 이쪽이 오히려 학습 가능한 파라미터화 시키는 것이 아닐까? 아닌가? 해 봐야 안다.
# from temp_from_GRU import TP_encoder_MIT as TP


# import torchmetrics.functional as TF # 이걸로 메트릭 한번에 간편하게 할 수 있다던데?

# Cuda 써야겠지?
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # GPU 번호별로 0번부터 나열
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # 일단 원석이가 0, 1번 쓰고 있다 하니 2번으로 지정
device = "cuda" if torch.cuda.is_available() else "cpu" # 연산에 GPU 쓰도록 지정
print("Device :" + device) # 확인용
# input() # 일시정지용


# 하이퍼파라미터와 사전 설정값들은 모두 .json 파일에 집어넣도록 한다.

# json 읽어다가 반환(파일경로 없으면 에러띄우기)
def loadJson() : 
    if (len(sys.argv) != 2) : 
        print("config.json 파일 경로가 없거나 그 이상의 인자가 들어갔습니다!", len(sys.argv))
        exit()
    else : 
        with open(sys.argv[1], 'r') as f:
            print("config.json파일 읽기 성공!")
            return json.load(f)
        
# 파일 읽어들이고 변수들 할당하기
json_data = loadJson()
num_classes = json_data['num_classes']
num_encoders = json_data['num_encoders']
early_stop = json_data['early_stop']
learning_rate = json_data['init_lr']
batch_size = json_data['batch_size']
num_epochs = json_data['num_epochs']
train_path = json_data['train_path']
test_path = json_data['test_path']
class_weight = json_data['class_weight']


# 일단은 텐서보드 그대로 사용
# 텐서보드 선언(인자도 미리 뽑아두기; 나중에 json으로 바꿀 것!)
# 텐서보드 사용 유무를 json에서 설정하는 경우 눈치껏 조건문으로 비활성화!
board_class = 'binary' if num_classes == 2 else 'multi' # 클래스갯수를 1로 두진 않겠지?
# writer = SummaryWriter(log_dir="./tensorboard/"+"SNN_MLP_" + board_class + "_encoders" + str(num_encoders) + "_early" + str(early_stop) + "_lr" + str(learning_rate))



# 참고 : 이것 외에도 에포크, Loss까지 찍어야 하니 참고할 것!
earlystop_counter = early_stop
min_valid_loss = float('inf')
final_epoch = 0 # 마지막에 최종 에포크 확인용


# 이제 메인으로 사용할 SNN 모델이 들어간다 : 얘 안에 단일뉴런 인코딩하는녀석이 들어가는 것.
# 일단 spikingjelly에서 그대로 긁어왔으므로, 구동이 안되겠다 싶은 녀석들은 읽고 바꿔둘 것.
class SNN_MLP(nn.Module):
    def __init__(self, num_classes, num_encoders):
        super().__init__()

        # SNN TP인코더 : 근데 이제 기존의 Linear 레이어 있는걸로 적절히 주물러서 쓰기?
        self.encoders = nn.Sequential(
            # layer.Flatten(), # 어차피 1차원 데이터인데 필요없지 않나?
            layer.Linear(1, num_encoders), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            )

        # SNN 리니어 : 인코더 정상동작 확인용이니 여기선 안쓴다.
        
    def forward(self, x: torch.Tensor):
        return self.encoders(x)





# 데이터 가져오기용 클래스
# 추가 : 기존 데이터셋 이용할거면 기존꺼 써도 되지 않나?
class MITLoader_MLP_binary(Dataset):

    def __init__(self, csv_file, transforms: Callable = lambda x: x) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file).values
        self.transforms = transforms

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, item):
        signal = self.annotations[item, :-1]
        signal = torch.from_numpy(signal).float()
        if self.transforms : 
            signal = self.transforms(signal)
        
        label = int(self.annotations[item, -1])
        if label > 0:
            label = 1  # 1 이상인 모든 값은 1로 변환(난 이진값 처리하니깐)

        return signal, torch.tensor(label).long()


# test 데이터로 정확도 측정 : 여기선 값 설정과 함께 1번의 순전파만 해서 신호가 잘 나오는지를 봐야 한다.
def check_accuracy(loader, model):

    # 각종 메트릭들 리셋(train에서 에폭마다 돌리므로 얘도 에폭마다 들어감)
    total_loss = 0

    # 모델 평가용으로 전환
    model.eval()
    
    print("validation 진행중...")

    with  torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            
            label_onehot = F.one_hot(y, num_classes).float() # 원핫으로 MSE loss 쓸거임
            
            # 순전파 : SNN용으로 바꿔야 함
            timestep = x.shape[1] # SNN은 타임스텝이 필요함
            
            out_fr = 0. # 출력 발화빈도를 이렇게 설정해두고, 나중에 출력인 리스트 형태로 더해진다 함
            # out_fr_list = []  # 출력 발화빈도를 리스트로 저장
            for t in range(timestep) : 
                timestep_data = x[:, t].unsqueeze(1)  # 각 timestep마다 (batch_size, 1) 크기로 자름
                out_fr += model(timestep_data) # 1회 순전파
        
        
            out_fr = out_fr / timestep
            # out_fr = torch.stack(out_fr_list).mean(dim=0)  # 타임스텝별 출력을 평균내어 합침
            
            loss = F.mse_loss(out_fr, label_onehot, reduction='none')

            weighted_loss = loss * class_weight[targets].unsqueeze(1) # 가중치 곱하기 : 여긴 배치 없는데 혹시 모르니..?
            final_loss = weighted_loss.mean() # 요소별 뭐시기 loss를 평균내서 전체 loss 계산?
            
            # 여기에도 total loss 찍기
            total_loss += final_loss.item()
            
            # 얘도 SNN 모델이니 초기화 필요
            functional.reset_net(model)



    # 모델 다시 훈련으로 전환
    model.train()






########### 학습시작! ############

# raw 데이터셋 가져오기
train_dataset = MITLoader_MLP_binary(csv_file=train_path)
test_dataset = MITLoader_MLP_binary(csv_file=test_path)

# 랜덤노이즈, 랜덤쉬프트는 일단 여기에 적어두기만 하고 구현은 미뤄두자.

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # 물론 이건 그대로 써도 될 듯?
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 데이터셋의 클래스 당 비율 불일치 문제가 있으므로 가중치를 통해 균형을 맞추도록 한다.
class_weight = torch.tensor(class_weight, device=device)


# SNN 네트워크 초기화
model = SNN_MLP(num_encoders=num_encoders, num_classes=num_classes).to(device=device)


# Loss와 optimizer, scheduler (클래스별 배율 설정 포함)
pos_weight = torch.tensor([0.2, 0.8]).to(device)
criterion = nn.CrossEntropyLoss(weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=0.00001)

# inplace 오류해결을 위해 이상위치 찾기
torch.autograd.set_detect_anomaly(True)

# 모델 학습 시작(학습추이 확인해야 하니 훈련, 평가 모두 Acc, F1, AUROC, AUPRC 넣을 것!)
# GRU가 아니기 때문에 해당하는 부분은 바꿔둬야 할 것으로 보임..
# # 이거 다 째도 될듯?
# for epoch in range(num_epochs):
#     # 에폭마다 각종 메트릭들 리셋
#     total_loss = 0


#     # 배치단위 실행
#     for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
#         # 데이터 cuda에 갖다박기
#         data = data.to(device=device).squeeze(1) # 일차원이 있으면 제거, 따라서 batch는 절대 1로 두면 안될듯
#         targets = targets.to(device=device)
        
        
#         # print(f"Batch {batch_idx}, targets: {targets}") # 라벨 잘 들어갔는지 확인용
#         label_onehot = F.one_hot(targets, num_classes).float() # 원핫으로 MSE loss 쓸거임
#         # print("onehot :", label_onehot) # 원핫도 잘나옴

#         # 순전파
#         # scores = model(data)
#         # loss = criterion(scores, targets)
#         # total_loss += loss.item() # loss값 따오기
        
        
#         # 순전파 : SNN용으로 바꿔야 함
#         timestep = data.shape[1] # SNN은 타임스텝이 필요함
#         out_fr = 0. # 출력 발화빈도를 이렇게 설정해두고, 나중에 출력인 리스트 형태로 더해진다 함
#         for t in range(timestep) :  # 원래 timestep 들어가야함
#             timestep_data = data[:, t].unsqueeze(1)  # 각 timestep마다 (batch_size, 1) 크기로 자름
#             out_fr += model(timestep_data) # 1회 순전파
        
        
#         out_fr = out_fr / timestep
#         loss = F.mse_loss(out_fr, label_onehot, reduction='none') # 요소별로 loss를 구해야 해서 reduction을 넣는다는데..
#         weighted_loss = loss * class_weight[targets].unsqueeze(1) # 가중치 곱하고 배치 차원 확장
#         final_loss = weighted_loss.mean() # 요소별 뭐시기 loss를 평균내서 전체 loss 계산?


#         # 얘도 일단 total_loss를 찍어봐야..겠지?
#         total_loss += final_loss.item()

#         # 역전파
#         optimizer.zero_grad()
#         final_loss.backward(retain_graph=True)

#         # 아담 옵티머스 프라임 출격
#         optimizer.step()

        
#         # SNN : 모델 초기화
#         functional.reset_net(model)


#     # 스케줄러는 에포크 단위로 진행
#     scheduler.step()

#     # 한 에포크 진행 다 됐으면 training 지표 tensorboard에 찍고 valid 돌리기
#     train_loss = total_loss / len(train_loader)


#     valid_loss = check_accuracy(test_loader, model) # valid(자체적으로 tensorboard 내장됨), 반환값으로 얼리스탑 확인하기

#     print('epoch ' + str(epoch) + ', valid loss : ' + str(valid_loss))

#     if valid_loss < min_valid_loss : 
#         min_valid_loss = valid_loss
#         earlystop_counter = early_stop
#     else : 
#         earlystop_counter -= 1
#         if earlystop_counter == 0 : 
#             final_epoch = epoch
#             break # train epoch를 빠져나옴

    

check_accuracy(test_loader, model)


print("training finished; epoch :" + str(final_epoch))