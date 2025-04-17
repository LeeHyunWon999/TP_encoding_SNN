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
from torch.utils.tensorboard import SummaryWriter # tensorboard 기록용
import time # 텐서보드 폴더명에 쓸 시각정보 기록용
import random # 랜덤시드 고정용

# 여긴 인코더 넣을때 혹시 몰라서 집어넣었음
import sys
import os
import json
import numpy as np

# 얘는 SNN 학습이니까 당연히 있어야겠지? 특히 SNN 모델을 따로 만드려는 경우엔 뉴런 말고도 넣을 것이 많다.
# import spikingjelly.activation_based as jelly
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

from sklearn.model_selection import KFold # cross-validation용

class trainer : 
    def __init__(self, args) -> None: 
        self.args = args

        # cuda 환경 사용
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # GPU 번호별로 0번부터 나열
        os.environ["CUDA_VISIBLE_DEVICES"]= str(args['device']['gpu'])  # 상황에 맞춰 변경할 것
        device = "cuda" if torch.cuda.is_available() else "cpu" # 연산에 GPU 쓰도록 지정
        print("Device :" + device) # 확인용

        # 랜덤시드 고정
        seed = args['executor']['args']['random_seed']
        deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 파일 실행 기준 시간 변수 생성
        exec_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        # 텐서보드에 찍을 메트릭 여기서 정의
        f1_micro = torchmetrics.F1Score(num_classes=2, average='micro', task='binary').to(device)
        f1_weighted = torchmetrics.F1Score(num_classes=2, average='weighted', task='binary').to(device)
        auroc_macro = torchmetrics.AUROC(num_classes=2, average='macro', task='binary').to(device)
        auroc_weighted = torchmetrics.AUROC(num_classes=2, average='weighted', task='binary').to(device)
        auprc = torchmetrics.AveragePrecision(num_classes=2, task='binary').to(device)
        accuracy = torchmetrics.Accuracy(threshold=0.5, task='binary').to(device)

        # 참고 : 이것 외에도 에포크, Loss까지 찍어야 하니 참고할 것!
        earlystop_counter = args['executor']['args']['early_stop_epoch']
        min_valid_loss = float('inf')
        final_epoch = 0 # 마지막에 최종 에포크 확인용
    

    # 훈련 작업(k-fold로 진행)
    def train() : 
        pass









    # 검증 작업(validation), 테스트와 별개로 epoch당 1회씩 진행하기 (훈련 메소드 완성 후 진행하기)
    def valid(self, loader, model, writer):

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
        
        print("validation 진행중...")

        with  torch.no_grad():
            for x, y in loader:         ############### train쪽에서 코드 복붙 시 (data, targets) 가 (x, y) 로 바뀌는 것에 유의할 것!!!!!!!!###############
                x = x.to(device=device).squeeze(1)
                y = y.to(device=device)
                
                label_onehot = F.one_hot(y, num_classes).float() # 원핫으로 MSE loss 쓸거임
                
                out_fr = 0. # 출력 발화빈도를 이렇게 설정해두고, 나중에 출력인 리스트 형태로 더해진다 함


                # 필터연산 (타임스텝은 안에 들어가있음)
                out_fr = model(x) # 앞으로도 그렇겠지만, 순전파꺼 넣는다고 x 말고 data 넣는 치명적 실수 하지 말 것 !!!
            


                    
                    
                # 여기부턴 출력값 처리에 관한 내용이니 메커니즘 건들거면 여긴 안만져도 됨
                
                # out_fr = out_fr / timestep # 발화비율은 CNN필터 모델 안에서 계산되어 나온다.
                # out_fr = torch.stack(out_fr_list).mean(dim=0)  # 타임스텝별 출력을 평균내어 합침
                
                loss = F.mse_loss(out_fr, label_onehot, reduction='none')

                weighted_loss = loss * class_weight[y].unsqueeze(1) # 가중치 곱하기 : 여긴 배치 없는데 혹시 모르니..?
                final_loss = weighted_loss.mean() # 요소별 뭐시기 loss를 평균내서 전체 loss 계산?
                
                # 여기에도 total loss 찍기
                total_loss += final_loss.item()

                # 여기도 메트릭 update해야 compute 가능함
                # 여기도 마찬가지로 크로스엔트로피 드가는거 생각해서 1차원으로 변경 필요함
                preds = torch.argmax(out_fr, dim=1)
                accuracy.update(preds, y)
                f1_micro.update(preds, y)
                f1_weighted.update(preds, y)
                auroc_macro.update(preds, y)
                auroc_weighted.update(preds, y)
                probabilities = F.softmax(out_fr, dim=1)[:, 1]  # 클래스 "1"의 확률 추출
                auprc.update(probabilities, y)
                
                # 얘도 SNN 모델이니 초기화 필요
                functional.reset_net(model)

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