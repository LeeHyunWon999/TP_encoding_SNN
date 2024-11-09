# LIF 포아송 : 가중치 제약 없이 학습된 poisson의 weight, signal 히스토그램 찍기



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
from torch.utils.tensorboard import SummaryWriter # tensorboard 기록용
import time # 텐서보드 폴더명에 쓸 시각정보 기록용
import random # 랜덤시드 고정용

# 여긴 인코더 넣을때 혹시 몰라서 집어넣었음
import sys
import os
import json
import numpy as np

# 얘는 SNN 학습이니까 당연히 있어야겠지?
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

# 히스토그램 찍기 위한 플롯
import matplotlib.pyplot as plt



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
cuda_gpu = json_data['cuda_gpu']
model_name = json_data['model_name']
num_classes = json_data['num_classes']
num_encoders = json_data['num_encoders'] # 편의상 이녀석을 MIT-BIH 길이인 187로 지정하도록 한다.
early_stop = json_data['early_stop']
early_stop_enable = json_data['early_stop_enable']
learning_rate = json_data['init_lr']
batch_size = json_data['batch_size']
num_epochs = json_data['num_epochs']
train_path = json_data['train_path']
test_path = json_data['test_path']
class_weight = json_data['class_weight']
encoder_min = json_data['encoder_min']
encoder_max = json_data['encoder_max']
hidden_size = json_data['hidden_size']
hidden_size_2 = json_data['hidden_size_2']
scheduler_tmax = json_data['scheduler_tmax']
scheduler_eta_min = json_data['scheduler_eta_min']
encoder_requires_grad = json_data['encoder_requires_grad']
timestep = json_data['timestep']
random_seed = json_data['random_seed']
checkpoint_save = json_data['checkpoint_save']
checkpoint_path = json_data['checkpoint_path']
threshold_value = json_data['threshold_value']
leak_decay = json_data['leak_decay']
need_bias = json_data['need_bias']
saved_model_path = json_data['saved_model_path'] # 이게 여기서 따로 추가됨

# Cuda 써야겠지?
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # GPU 번호별로 0번부터 나열
os.environ["CUDA_VISIBLE_DEVICES"]= str(cuda_gpu)  # 상황에 맞춰 변경할 것
device = "cuda" if torch.cuda.is_available() else "cpu" # 연산에 GPU 쓰도록 지정
print("Device :" + device) # 확인용
# input() # 일시정지용


# 랜덤시드 고정
seed = random_seed
deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False



# 학습 코드의 모델 그대로 가져오기
class SNN_MLP(nn.Module):
    def __init__(self, num_classes, num_encoders, hidden_size, leak, threshold_value, bias_option):
        super().__init__()
        
        # SNN 리니어 : 인코더 입력 -> 히든
        self.hidden = nn.Sequential(
            layer.Linear(num_encoders, hidden_size, bias = bias_option), # bias는 일단 기본값 True로 두기
            neuron.LIFNode(surrogate_function=surrogate.ATan(), v_reset=0.0, v_threshold=threshold_value, tau=leak, decay_input=False),
            )

        # SNN 리니어 : 히든 -> 출력
        self.layer = nn.Sequential(
            layer.Linear(hidden_size, num_classes, bias = bias_option), # bias는 일단 기본값 True로 두기
            neuron.LIFNode(surrogate_function=surrogate.ATan(), v_reset=0.0, v_threshold=threshold_value, tau=leak, decay_input=False),
            )
        
    
    # 여기서 인코딩 레이어만 딱 빼면 된다.
    def forward(self, x: torch.Tensor):
        x = self.hidden(x)
        return self.layer(x)
    
    
# 데이터 가져오는 알맹이 클래스
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




# 모델 로드, 가중치도 같이 진행
model = SNN_MLP(num_encoders=num_encoders, num_classes=num_classes, hidden_size=hidden_size, 
                leak=leak_decay, threshold_value=threshold_value, bias_option=need_bias).to(device=device)
checkpoint = torch.load(saved_model_path)
model.load_state_dict(checkpoint["model_state_dict"])

# 데이터도 여기서 로드
test_dataset = MITLoader_MLP_binary(csv_file=test_path)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 인코더 지정 : 여긴 포아송
encoder = encoding.PoissonEncoder()


# 가중치 히스토그램 생성
def save_weights_histogram(model, x_min = None, x_max = None, file_path="default_weights_histogram.png"):
    weights = []
    for name, param in model.named_parameters():
        if "weight" in name:
            weights.extend(param.detach().cpu().numpy().flatten())
    
    plt.figure()
    if x_min is not None and x_max is not None:
            plt.hist(weights, bins=100, range=(x_min, x_max), color="blue", alpha=0.7)  # 범위 설정
    else:
        plt.hist(weights, bins=100, color="green", alpha=0.7)
    plt.title("Weights Histogram")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.savefig(file_path)  # 플롯을 파일로 저장
    plt.close()  # 메모리 해제를 위해 플롯 닫기




# 순전파 중 LIF layer 직전의 signal 히스토그램 생성
def save_signals_histogram(model, x_min = None, x_max = None, file_path="default_signals_histogram.png"):
    signals = []

    def hook(module, input, output):
        # input[0]의 데이터 형식을 명시적으로 변환하여 1차원 배열로 저장
        if isinstance(input, tuple):
            input_signal = input[0].detach().cpu().numpy()
            signals.extend(input_signal.flatten())  # flatten하여 1차원 리스트로 저장

    # Hook 등록
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, neuron.LIFNode):  # SpikingJelly의 LIFNode
            hooks.append(module.register_forward_hook(hook))
    
    # Forward 실행 : 모델마다 이 부분도 다르게 해야 할 것
    model.eval()
    with torch.no_grad():
        # 1배치만 시도(256)
        for x, y in tqdm(test_loader, desc="Running inference"):
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            
            for t in range(timestep): 
                encoded_data = encoder(x)  # Poisson encoding
                _ = model(encoded_data)  # 모델 순전파
            
            functional.reset_net(model)  # 배치마다 모델 초기화
            break

    # Hook 해제
    for h in hooks:
        h.remove()
        
    print(str(len(signals)) + "만큼의 신호 수집 완료.")

    # 플롯 저장
    # 플롯 저장
    if len(signals) > 0:
        print(f"Collected {len(signals)} signal values. Saving histogram...")
        plt.figure()
        if x_min is not None and x_max is not None:
            plt.hist(signals, bins=100, range=(x_min, x_max), color="green", alpha=0.7)  # 범위 설정
        else:
            plt.hist(signals, bins=100, color="green", alpha=0.7)
        plt.title("Signals Histogram (Before LIFNode)")
        plt.xlabel("Signal Value")
        plt.ylabel("Frequency")
        plt.xlim(x_min, x_max)  # x축 범위 조정
        plt.savefig(file_path)  # 플롯을 파일로 저장
        plt.close()  # 메모리 해제를 위해 플롯 닫기
    else:
        print("No signal values collected. Check the hook or model behavior.")
    



# 가중치 히스토그램 저장
save_weights_histogram(model, x_min = -3, x_max = 3, file_path = model_name + "_weights_histogram.png")


# 신호 히스토그램 저장
save_signals_histogram(model, x_min = -40, x_max = 40, file_path = model_name + "_signals_histogram.png")
