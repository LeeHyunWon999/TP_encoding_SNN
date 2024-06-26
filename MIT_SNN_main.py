# 시작 파일.
# 학습 시작 전에 json 파일 이용해서 하이퍼파라미터와 함께 집어넣고, 데이터로더 이용해서 지정된 횟수만큼 학습 지시하며 필요한 경우 텐서보드에 찍는다.
# nn을 상속하는 어떤 녀석이든지 딥러닝 계열로 들어갈 수 있다면, 기존의 딥러닝 MLP 모델의 구조를 그대로 가져오고 모델만 SNN으로 바꾸는 식으로 구성해볼까?
# 일단 인코더 모델 따로 만들어진건 그대로 두고, SNN 모델 안에 그걸 넣으며, 연산은 텐서에서 진행하도록 하고 해당하는 함수만 뗴와다가 SNN 모델에 넣도록 해본다.

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
from torch.utils.tensorboard import SummaryWriter # tensorboard 기록용

# 여긴 인코더 넣을때 혹시 몰라서 집어넣었음
import sys
import os
import json
import numpy as np

# 얘는 SNN 학습이니까 당연히 있어야겠지? 특히 SNN 모델을 따로 만드려는 경우엔 뉴런 말고도 넣을 것이 많다.
import spikingjelly.activation_based as jelly
# from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

# 이번엔 인코더도 올려둔다.
from temp_from_GRU import TP_encoder_MIT as TP


# import torchmetrics.functional as TF # 이걸로 메트릭 한번에 간편하게 할 수 있다던데?

# Cuda 써야겠지?
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # GPU 번호별로 0번부터 나열
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # 일단 원석이가 0, 1번 쓰고 있다 하니 2번으로 지정
device = "cuda" if torch.cuda.is_available() else "cpu" # 연산에 GPU 쓰도록 지정
print("Device :" + device) # 확인용
# input() # 일시정지용


# 하이퍼파라미터와 사전 설정값들은 모두 .json 파일에 집어넣도록 한다. 물론 여기선 변수 할당.



# 일단은 텐서보드 그대로 사용
# 텐서보드 선언(인자도 미리 뽑아두기; 나중에 json으로 바꿀 것!)
# 텐서보드 사용 유무를 json에서 설정하는 경우 눈치껏 조건문으로 비활성화!
board_class = 'binary' if num_classes == 2 else 'multi' # 클래스갯수를 1로 두진 않겠지?
writer = SummaryWriter(log_dir="./tensorboard/"+"GRU_binary" + board_class + "_input" + str(input_size) + "_early" + str(early_stop) + "_lr" + str(learning_rate))

# 텐서보드에 찍을 메트릭 여기서 정의
f1_micro = torchmetrics.F1Score(num_classes=2, average='micro', task='binary').to(device)
f1_weighted = torchmetrics.F1Score(num_classes=2, average='weighted', task='binary').to(device)
auroc_macro = torchmetrics.AUROC(num_classes=2, average='macro', task='binary').to(device)
auroc_weighted = torchmetrics.AUROC(num_classes=2, average='weighted', task='binary').to(device)
auprc = torchmetrics.AveragePrecision(num_classes=2, task='binary').to(device)
accuracy = torchmetrics.Accuracy(threshold=0.5, task='binary').to(device)

# 참고 : 이것 외에도 에포크, Loss까지 찍어야 하니 참고할 것!
earlystop_counter = early_stop
min_valid_loss = float('inf')
final_epoch = 0 # 마지막에 최종 에포크 확인용


# 이제 메인으로 사용할 SNN 모델이 들어간다 : 얘 안에 단일뉴런 인코딩하는녀석이 들어가는 것.
# 일단 spikingjelly에서 그대로 긁어왔으므로, 구동이 안되겠다 싶은 녀석들은 읽고 바꿔둘 것.
class SNN_binary(nn.Module):
    def __init__(self, tau):
        super().__init__()

        # SNN TP인코더(인자 수정해야 함!)
        self.encoders = nn.ModuleList([TP.TP_neuron()])수정수정

        # SNN 리니어
        self.layer = nn.Sequential(
            jelly.layer.Flatten(),
            jelly.layer.Linear(187 * 10, 2), # bias는 일단 기본값 True로 두기
            jelly.neuron.IFNode(tau=tau, surrogate_function=jelly.surrogate.ATan()),
            )

    def forward(self, x: torch.Tensor):
        return self.layer(x)





# 데이터 가져오기용 클래스(아마 여길 가장 많이 바꿔야 할 듯,,,)
# MIT-BIH를 인코딩해야 하므로, 공간 많이 잡아먹지 싶다. 이건 /data/leehyunwon/ 이쪽에 변환 후 넣고 나서 여기서 불러오는 식으로 해야 할 듯
# 나중에 json으로 옮기면서 동시에 데이터로더에 넣었던 인코딩 레이어를 GRU 쪽에 붙여서 학습 가능하게끔 만들기...? 되는지조차 불투명하니 나중에 생각해봐야 할 듯

# 커스텀 데이터셋 관리 클래스
class MITLoader(Dataset):

    def __init__(self, original_csv, encoded_npy, transforms: None) -> None:
        super().__init__()
        self.annotations = pd.read_csv(original_csv).values # MIT 라벨 읽기용
        self.encoded = np.load(encoded_npy) # MIT 인코딩된 데이터 로딩용
        self.transforms = transforms
        
        # 근데 이제 RNN은 입력 텐서의 차원을 (시퀀스, 배치, 입력크기) 로 기대하므로, 원본 인코딩 데이터의 (데이터, 입력크기(뉴런갯수), 시퀀스) 를 변형해야 한다.
        # 참고로 배치는 나중에 추가되는거니까 크게 신경 안써도 되고, 데이터도 어차피 인덱스별로 날아가므로 시퀀스와 입력크기 순서를 바꾸도록 한다.
        self.encoded = np.transpose(self.encoded, (0, 2, 1))

    def __len__(self):
        return self.annotations.shape[0] # shape은 차원당 요소 갯수를 튜플로 반환하므로 행에 해당하는 0번 값 반환 : 이것도 혹시 모르니 변환된 데이터에 대한 걸로 바꿀까? 걍 냅둘까?

    def __getitem__(self, item):
        signal = self.encoded[item, :-1] # 마지막은 라벨이니 그거 빼고 집어넣기
    
        # numpy 배열을 텐서로 변경
        signal = torch.from_numpy(signal).float()
        
        # transform이 있는 경우 적용
        if self.transforms:
            signal = self.transforms(signal)
            
        # 라벨 변경 : 이진 분류를 할 것이므로 0인 경우 0, 아니면 1로 바꿔야 함
        label = int(self.annotations[item, -1])
        if label > 0:
            label = 1  # 1 이상인 모든 값은 1로 변환(난 이진값 처리하니깐)
            
        label = torch.tensor(label, dtype=torch.long) # 처리 다 된 라벨을 텐서로 변환

        return signal, label


# test 데이터로 정확도 측정
def check_accuracy(loader, model):

    # 각종 메트릭들 리셋(train에서 에폭마다 돌리므로 얘도 에폭마다 들어감)
    total_loss = 0
    accuracy.reset()
    f1_micro.reset()
    f1_weighted.reset()
    auroc_macro.reset()
    auroc_weighted.reset()
    auprc.reset()

    # 모델 평가용으로 전환
    model.eval()

    with  torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            loss = criterion(scores, y) # loss값 직접 따오기 위해 여기서도 loss값 추려내기
            total_loss += loss.item() # loss값 따오기

            # 여기도 메트릭 update해야 compute 가능함
            # 여기도 마찬가지로 크로스엔트로피 드가는거 생각해서 1차원으로 변경 필요함
            preds = torch.argmax(scores, dim=1)
            accuracy.update(preds, y)
            f1_micro.update(preds, y)
            f1_weighted.update(preds, y)
            auroc_macro.update(preds, y)
            auroc_weighted.update(preds, y)
            probabilities = F.softmax(scores, dim=1)[:, 1]  # 클래스 "1"의 확률 추출
            auprc.update(probabilities, y)

    # 각종 평가수치들 만들고 tensorboard에 기록
    valid_loss = total_loss / len(loader)
    valid_accuracy = accuracy.compute()
    valid_f1_micro = f1_micro.compute()
    valid_f1_weighted = f1_weighted.compute()
    valid_auroc_macro = auroc_macro.compute()
    valid_auroc_weighted = auroc_weighted.compute()
    valid_auprc = auprc.compute()

    writer.add_scalar('valid_Loss', valid_loss, epoch)
    writer.add_scalar('valid_Accuracy', valid_accuracy, epoch)
    writer.add_scalar('valid_F1_micro', valid_f1_micro, epoch)
    writer.add_scalar('valid_F1_weighted', valid_f1_weighted, epoch)
    writer.add_scalar('valid_AUROC_macro', valid_auroc_macro, epoch)
    writer.add_scalar('valid_AUROC_weighted', valid_auroc_weighted, epoch)
    writer.add_scalar('valid_auprc', valid_auprc, epoch)

    # 모델 다시 훈련으로 전환
    model.train()

    # valid loss를 반환한다. 이걸로 early stop 확인.
    return valid_loss






# 학습시작!


# 일단 raw 데이터셋 가져오기
train_dataset = MITLoader(original_csv=train_path, encoded_npy=train_encoded_path, transforms=None)
test_dataset = MITLoader(original_csv=test_path, encoded_npy=test_encoded_path, transforms=None)

# 랜덤노이즈, 랜덤쉬프트는 일단 여기에 적어두기만 하고 구현은 미뤄두자.

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # 물론 이건 그대로 써도 될 듯?
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# SNN 네트워크 초기화



# Loss와 optimizer, scheduler (클래스별 배율 설정 포함)
pos_weight = torch.tensor([0.2, 0.8]).to(device)
criterion = nn.CrossEntropyLoss(weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=0.00001)



# 모델 학습 시작(학습추이 확인해야 하니 훈련, 평가 모두 Acc, F1, AUROC, AUPRC 넣을 것!)
# GRU가 아니기 때문에 해당하는 부분은 바꿔둬야 할 것으로 보임..
for epoch in range(num_epochs):
    # 에폭마다 각종 메트릭들 리셋
    total_loss = 0
    accuracy.reset()
    f1_micro.reset()
    f1_weighted.reset()
    auroc_macro.reset()
    auroc_weighted.reset()
    auprc.reset()


    # 배치단위 실행
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # 데이터 cuda에 갖다박기
        data = data.to(device=device).squeeze(1) # 일차원이 있으면 제거, 따라서 batch는 절대 1로 두면 안될듯
        targets = targets.to(device=device)

        # 순전파
        scores = model(data)
        loss = criterion(scores, targets)

        total_loss += loss.item() # loss값 따오기

        # 역전파
        optimizer.zero_grad()
        loss.backward()

        # 아담 옵티머스 프라임 출격
        optimizer.step()

        # 배치마다 각 메트릭을 업데이트한 뒤에 compute해야 한댄다
        # 근데 이제 scores는 크로스엔트로피로, 원시 클래스 확률 로짓이 그대로 있으므로 argmax를 시켜서 값 하나를 뽑아야 한다.
        preds = torch.argmax(scores, dim=1)
        accuracy.update(preds, targets)
        f1_micro.update(preds, targets)
        f1_weighted.update(preds, targets)
        auroc_macro.update(preds, targets)
        auroc_weighted.update(preds, targets)
        probabilities = F.softmax(scores, dim=1)[:, 1]  # 클래스 "1"의 확률 추출
        auprc.update(probabilities, targets)


    # 스케줄러는 에포크 단위로 진행
    scheduler.step()

    # 한 에포크 진행 다 됐으면 training 지표 tensorboard에 찍고 valid 돌리기
    train_loss = total_loss / len(train_loader)
    train_accuracy = accuracy.compute()
    train_f1_micro = f1_micro.compute()
    train_f1_weighted = f1_weighted.compute()
    train_auroc_macro = auroc_macro.compute()
    train_auroc_weighted = auroc_weighted.compute()
    train_auprc = auprc.compute()

    writer.add_scalar('train_Loss', train_loss, epoch)
    writer.add_scalar('train_Accuracy', train_accuracy, epoch)
    writer.add_scalar('train_F1_micro', train_f1_micro, epoch)
    writer.add_scalar('train_F1_weighted', train_f1_weighted, epoch)
    writer.add_scalar('train_AUROC_macro', train_auroc_macro, epoch)
    writer.add_scalar('train_AUROC_weighted', train_auroc_weighted, epoch)
    writer.add_scalar('train_auprc', train_auprc, epoch)

    valid_loss = check_accuracy(test_loader, model) # valid(자체적으로 tensorboard 내장됨), 반환값으로 얼리스탑 확인하기

    print('epoch ' + str(epoch) + ', valid loss : ' + str(valid_loss))

    if valid_loss < min_valid_loss : 
        min_valid_loss = valid_loss
        earlystop_counter = early_stop
    else : 
        earlystop_counter -= 1
        if earlystop_counter == 0 : 
            final_epoch = epoch
            break # train epoch를 빠져나옴
    
    




print("training finished; epoch :" + str(final_epoch))


# 마지막엔 텐서보드 닫기(기록유무도 json에 적는 경우 눈치껏 비활성화)
writer.close()