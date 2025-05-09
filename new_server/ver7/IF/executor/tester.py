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

import sys
import json

# 얘는 SNN 학습이니까 당연히 있어야겠지? 특히 SNN 모델을 따로 만드려는 경우엔 뉴런 말고도 넣을 것이 많다.
# import spikingjelly.activation_based as jelly
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from sklearn.model_selection import KFold # cross-validation용

from util import util

class tester : 
    def __init__(self, args) -> None: 
        self.args = args

        # cuda 환경 사용
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # GPU 번호별로 0번부터 나열
        os.environ["CUDA_VISIBLE_DEVICES"]= str(args['device']['gpu'])  # 상황에 맞춰 변경할 것
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # 연산에 GPU 쓰도록 지정
        print("Device :" + self.device) # 확인용

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

        # 파일 실행 기준 시간 변수 생성 (아마 안 쓸 것)
        # self.exec_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        # 텐서보드에 찍을 메트릭 여기서 정의
        self.f1_micro = torchmetrics.F1Score(num_classes=args['model']['args']['num_classes'], average='micro', task="multiclass").to(self.device)
        self.f1_weighted = torchmetrics.F1Score(num_classes=args['model']['args']['num_classes'], average='weighted', task="multiclass").to(self.device)
        self.auroc_macro = torchmetrics.AUROC(num_classes=args['model']['args']['num_classes'], average='macro', task="multiclass").to(self.device)
        self.auroc_weighted = torchmetrics.AUROC(num_classes=args['model']['args']['num_classes'], average='weighted', task="multiclass").to(self.device)
        self.auprc = torchmetrics.AveragePrecision(num_classes=args['model']['args']['num_classes'], task="multiclass").to(self.device)
        self.accuracy = torchmetrics.Accuracy(num_classes=args['model']['args']['num_classes'], task="multiclass").to(self.device)

        # 참고 : 이것 외에도 에포크, Loss까지 찍어야 하니 참고할 것!
        self.earlystop_counter = args['executor']['args']['early_stop_epoch']
        self.min_valid_loss = float('inf')
        self.final_epoch = 0 # 마지막에 최종 에포크 확인용

        # 가중치 비율 텐서로 옮기기
        self.class_weight = torch.tensor(args['loss']['weight'], device=self.device)
    

    # 테스트 진행 : fold만큼 반복, 모델에 가중치 넣고 돌리는 작업 수행
    def test(self) : 
        args = self.args
        
        # 데이터셋 로더 선정 (모델은 체크포인트를 위해 각 fold 안에서 선언하는 편이 나을 듯..?)
        test_dataset = util.get_data_loader_test(args['data_loader'])
        test_loader = DataLoader(test_dataset, batch_size=args['data_loader']['args']['batch_size'], 
                                      num_workers=args['data_loader']['args']['num_workers'],
                                      shuffle=True, drop_last=True)
        
        # exec_time_test : test용 exec_time, 기존의 시간 값으로부터 추출출
        exec_time_test = args['executor']['args']['checkpoint']['path'].split("_")[-3]

        # fold만큼 반복
        for i in args['executor']['args']['k_folds'] : 
            # tensorboard writer 설정
            writer = SummaryWriter(log_dir= args['tensorboard']['path'] + f"{args['model']['type']}" + "_" + args['data_loader']['type'] + "_" + 
                                   exec_time_test + f"_test{i + 1}")
            
            # model 로드
            model = ???
            checkpoint = torch.load(saved_model_dir)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()