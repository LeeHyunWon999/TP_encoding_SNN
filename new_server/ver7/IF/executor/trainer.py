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

class trainer : 
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

        # 파일 실행 기준 시간 변수 생성
        self.exec_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        # 텐서보드에 찍을 메트릭 여기서 정의
        self.f1_micro = torchmetrics.F1Score(num_classes=2, average='micro', task='binary').to(self.device)
        self.f1_weighted = torchmetrics.F1Score(num_classes=2, average='weighted', task='binary').to(self.device)
        self.auroc_macro = torchmetrics.AUROC(num_classes=2, average='macro', task='binary').to(self.device)
        self.auroc_weighted = torchmetrics.AUROC(num_classes=2, average='weighted', task='binary').to(self.device)
        self.auprc = torchmetrics.AveragePrecision(num_classes=2, task='binary').to(self.device)
        self.accuracy = torchmetrics.Accuracy(threshold=0.5, task='binary').to(self.device)

        # 참고 : 이것 외에도 에포크, Loss까지 찍어야 하니 참고할 것!
        self.earlystop_counter = args['executor']['args']['early_stop_epoch']
        self.min_valid_loss = float('inf')
        self.final_epoch = 0 # 마지막에 최종 에포크 확인용

        # 가중치 비율 텐서로 옮기기
        self.class_weight = torch.tensor(args['loss']['weight'], device=self.device)



    

    # 훈련 작업(k-fold로 진행)
    def train(self) : 
        args = self.args

        # 데이터셋 로더 선정 (모델은 각 fold 안에서 선언하는 편이 나을 듯..?)
        train_dataset = util.get_data_loader_train(args['data_loader'])

        # assert len(train_dataset) > 0, "🚨 CinC 데이터셋이 비어있습니다!"
        # print(f"✅ Dataset size: {len(train_dataset)}")

        # temp_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
        # print(f"✅ DataLoader batch count: {len(temp_train_loader)}")

        # k-fold 밑작업
        kf = KFold(n_splits=args['executor']['args']['k_folds'], shuffle=True, random_state=args['executor']['args']['random_seed'])

        # k-Fold 수행
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            print(f"Starting fold {fold + 1}/{args['executor']['args']['k_folds']}...")

            # Train/Validation 데이터셋 분리
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=args['data_loader']['args']['batch_size'], 
                                      num_workers=args['data_loader']['args']['num_workers'],
                                      shuffle=True, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=args['data_loader']['args']['batch_size'], 
                                    num_workers=args['data_loader']['args']['num_workers'],
                                    shuffle=False)

            # TensorBoard 폴더 설정
            writer = SummaryWriter(log_dir=f"./tensorboard/{args['model']['type']}" + "_" + args['data_loader']['type'] + "_" + 
                                   self.exec_time + f"_fold{fold + 1}")

            # 체크포인트 위치도 상세히 갱신
            checkpoint_path_fold = args['executor']['args']['checkpoint']['path'] + str(str(args['model']['type']) + "_" + args['data_loader']['type'] + "_" + 
                                                                                        self.exec_time + f"_fold{fold + 1}")
            json_output_fold = checkpoint_path_fold + "_config.json" # 체크포인트에 동봉되는 config 용
            lastpoint_path_fold = checkpoint_path_fold + "_lastEpoch.pt" # 최종에포크 저장용
            checkpoint_path_fold += ".pt" # 체크포인트 확장자 마무리
            
            # SNN 네트워크 초기화
            model = util.get_model(args['model'], device=self.device).to(device=self.device)

            # 옵티마이저, 스케줄러
            train_params = [p for p in model.parameters() if p.requires_grad] # 'requires_grad'가 False인 파라미터 말고 나머지는 학습용으로 돌리기기
            optimizer = util.get_optimizer(train_params, args['executor']['args']['optimizer'])
            scheduler = util.get_scheduler(optimizer, args['executor']['args']['scheduler'])

            # Training Loop
            for epoch in range(args['executor']['args']['num_epochs']):
                model.train()
                total_loss = 0
                self.accuracy.reset()
                self.f1_micro.reset()
                self.f1_weighted.reset()
                self.auroc_macro.reset()
                self.auroc_weighted.reset()
                self.auprc.reset()

                for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
                    data = data.to(device=self.device).squeeze(1) # 일차원이 있으면 제거, 따라서 batch는 절대 1로 두면 안될듯
                    targets = targets.to(device=self.device)
                    label_onehot = F.one_hot(targets, args['model']['args']['num_classes']).float() # 원핫으로 MSE loss 쓸거임

                    # 순전파
                    out_fr = util.propagation(model, data, args['model'])

                    
                    loss = util.get_loss(out_fr, label_onehot, args['loss'])

                    weighted_loss = loss * self.class_weight[targets].unsqueeze(1)
                    final_loss = weighted_loss.mean()
                    total_loss += final_loss.item()

                    # 역전파
                    optimizer.zero_grad()
                    final_loss.backward()
                    optimizer.step()

                    # 지표 계산
                    preds = torch.argmax(out_fr, dim=1)
                    self.accuracy.update(preds, targets)
                    self.f1_micro.update(preds, targets)
                    self.f1_weighted.update(preds, targets)
                    self.auroc_macro.update(preds, targets)
                    self.auroc_weighted.update(preds, targets)
                    probabilities = F.softmax(out_fr, dim=1)[:, 1]
                    self.auprc.update(probabilities, targets)

                    functional.reset_net(model)

                # 스케줄러는 에포크 단위로 진행
                scheduler.step()

                # 한 에포크 진행 다 됐으면 training 지표 tensorboard에 찍고 valid 돌리기
                train_loss = total_loss / len(train_loader)
                train_accuracy = self.accuracy.compute()
                train_f1_micro = self.f1_micro.compute()
                train_f1_weighted = self.f1_weighted.compute()
                train_auroc_macro = self.auroc_macro.compute()
                train_auroc_weighted = self.auroc_weighted.compute()
                train_auprc = self.auprc.compute()

                writer.add_scalar('train_Loss', train_loss, epoch)
                writer.add_scalar('train_Accuracy', train_accuracy, epoch)
                writer.add_scalar('train_F1_micro', train_f1_micro, epoch)
                writer.add_scalar('train_F1_weighted', train_f1_weighted, epoch)
                writer.add_scalar('train_AUROC_macro', train_auroc_macro, epoch)
                writer.add_scalar('train_AUROC_weighted', train_auroc_weighted, epoch)
                writer.add_scalar('train_auprc', train_auprc, epoch)

                # valid(자체적으로 tensorboard 내장됨), 반환값으로 얼리스탑 확인하기
                valid_loss = self.valid(val_loader, model, writer, epoch)

                print(f"Fold {fold + 1}, Epoch {epoch + 1}/{args['executor']['args']['num_epochs']}, Valid Loss: {valid_loss}")

                # 성능 좋게 나오면 체크포인트 저장 및 earlystop 갱신
                if args['executor']['args']['early_stop_enable'] :
                    if valid_loss < min_valid_loss : 
                        min_valid_loss = valid_loss
                        earlystop_counter = args['executor']['args']['early_stop_epoch']
                        if args['executor']['args']['checkpoint']['active'] : 
                            print("best performance, saving..")
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': valid_loss,
                                }, checkpoint_path_fold) # 가장 좋은 기록 나온 체크포인트 저장
                            with open(json_output_fold, "w", encoding='utf-8') as json_output : 
                                json.dump(args, json_output) # 설정파일도 저장
                    else : 
                        earlystop_counter -= 1
                        if earlystop_counter == 0 : # train epoch 빠져나오며 최종 모델 저장
                            final_epoch = epoch
                            print("last epoch model saving..")
                            torch.save({
                                'epoch': final_epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': valid_loss,
                                }, lastpoint_path_fold)
                            with open(json_output_fold, "w", encoding='utf-8') as json_output : 
                                json.dump(args, json_output) # 설정파일도 저장
                            break # train epoch를 빠져나옴
                else : 
                    final_epoch = epoch
                    if epoch == args['executor']['args']['num_epochs'] - 1 : # 얼리스탑과 별개로 최종 모델 저장
                        print("last epoch model saving..")
                        torch.save({
                            'epoch': final_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': valid_loss,
                            }, lastpoint_path_fold)
                        with open(json_output_fold, "w", encoding='utf-8') as json_output : 
                                json.dump(args, json_output) # 설정파일도 저장


            # 개별 텐서보드 닫기
            writer.close()

            print("fold " + f"{fold + 1}" + " training finished; epoch :" + str(final_epoch))

        print("All folds finished.")








    # 검증 작업(validation), 테스트와 별개로 epoch당 1회씩 진행하기 (훈련 메소드 완성 후 진행하기)
    def valid(self, loader, model, writer, epoch):
        args = self.args

        # 각종 메트릭들 리셋(train에서 에폭마다 돌리므로 얘도 에폭마다 들어감)
        total_loss = 0
        self.accuracy.reset()
        self.f1_micro.reset()
        self.f1_weighted.reset()
        self.auroc_macro.reset()
        self.auroc_weighted.reset()
        self.auprc.reset()

        # 모델 평가용으로 전환
        model.eval()
        
        print("validation 진행중...")

        with  torch.no_grad():
            for x, y in loader:         ############### train쪽에서 코드 복붙 시 (data, targets) 가 (x, y) 로 바뀌는 것에 유의할 것!!!!!!!!###############
                x = x.to(device=self.device).squeeze(1)
                y = y.to(device=self.device)
                
                label_onehot = F.one_hot(y, args['model']['args']['num_classes']).float() # 원핫으로 MSE loss 쓸거임
                
                # 순전파
                out_fr = util.propagation(model, x, args['model'])

                loss = util.get_loss(out_fr, label_onehot, args['loss'])

                weighted_loss = loss * self.class_weight[y].unsqueeze(1) # 가중치 곱하기
                final_loss = weighted_loss.mean() # 요소별 loss를 평균내서 전체 loss 계산
                
                # 여기에도 total loss 찍기
                total_loss += final_loss.item()

                # 여기도 메트릭 update해야 compute 가능함
                # 여기도 마찬가지로 크로스엔트로피 드가는거 생각해서 1차원으로 변경 필요함
                preds = torch.argmax(out_fr, dim=1)
                self.accuracy.update(preds, y)
                self.f1_micro.update(preds, y)
                self.f1_weighted.update(preds, y)
                self.auroc_macro.update(preds, y)
                self.auroc_weighted.update(preds, y)
                probabilities = F.softmax(out_fr, dim=1)[:, 1]  # 클래스 "1"의 확률 추출
                self.auprc.update(probabilities, y)
                
                # 얘도 SNN 모델이니 초기화 필요
                functional.reset_net(model)

        # 각종 평가수치들 만들고 tensorboard에 기록
        valid_loss = total_loss / len(loader)
        valid_accuracy = self.accuracy.compute()
        valid_f1_micro = self.f1_micro.compute()
        valid_f1_weighted = self.f1_weighted.compute()
        valid_auroc_macro = self.auroc_macro.compute()
        valid_auroc_weighted = self.auroc_weighted.compute()
        valid_auprc = self.auprc.compute()

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