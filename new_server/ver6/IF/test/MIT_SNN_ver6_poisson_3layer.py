# 포아송 학습 : 근데 이제 뒤쪽을 1024짜리 2층으로 쌓은.
# 히든레이어 집어넣는건 그대로 하되, 이번엔 아예 인코딩 레이어를 없애고 포아송 인코딩 함수만으로 정확도를 측정해보도록 한다.
# 구조가 좀 바뀌어야 할 것이다. 가령 타임스텝은 입력데이터 길이가 아닌 50으로 두는 등.

# 학습 시작 전에 json 파일 이용해서 하이퍼파라미터와 함께 집어넣고, 데이터로더 이용해서 지정된 횟수만큼 학습 지시하며 필요한 경우 텐서보드에 찍는다.

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

# 얘는 SNN 학습이니까 당연히 있어야겠지? 특히 SNN 모델을 따로 만드려는 경우엔 뉴런 말고도 넣을 것이 많다.
# import spikingjelly.activation_based as jelly
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

from sklearn.model_selection import KFold # cross-validation용

# 이쪽에선 SNN 모델을 넣지 않고, 바로 jelly.layer.Linear로 바로 들어가는 것을 시도해본다. 이쪽이 오히려 학습 가능한 파라미터화 시키는 것이 아닐까? 아닌가? 해 봐야 안다.
# from temp_from_GRU import TP_encoder_MIT as TP





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
reset_value_residual = json_data['reset_value_residual']
need_bias = json_data['need_bias']
k_folds = json_data['k_folds']
saved_model_dir = json_data['saved_model_dir']
temp_test_fold = json_data['temp_test_fold']

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
earlystop_counter = early_stop
min_valid_loss = float('inf')
final_epoch = 0 # 마지막에 최종 에포크 확인용

# 이제 메인으로 사용할 SNN 모델이 들어간다 : 포아송 인코딩이므로 인코딩 레이어 없앨 것!
# 일단 spikingjelly에서 그대로 긁어왔으므로, 구동이 안되겠다 싶은 녀석들은 읽고 바꿔둘 것.
class SNN_MLP(nn.Module):
    def __init__(self, num_classes, num_encoders, hidden_size, hidden_size_2, threshold_value, bias_option, reset_value_residual):
        super().__init__()
        
        # SNN 리니어 : 인코더 입력 -> 히든
        self.hidden = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(num_encoders, hidden_size, bias = bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )
        
        # SNN 리니어 : 인코더 히든 -> 히든2
        self.hidden2 = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size, hidden_size_2, bias = bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_reset=0.0, v_threshold=threshold_value),
            )

        # SNN 리니어 : 히든2 -> 출력
        self.layer = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size_2, num_classes, bias = bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )
        
    
    # 여기서 인코딩 레이어만 딱 빼면 된다.
    def forward(self, x: torch.Tensor):
        x = self.hidden(x)
        x = self.hidden2(x)
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


# test 데이터로 정확도 측정 : 얘도 훈련때랑 똑같이 집어넣어야 한다.
def check_accuracy(loader, model, writer):

    # 시간 측정
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

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
        start_event.record()  # 시작 이벤트 기록
        for x, y in loader:         ############### train쪽에서 코드 복붙 시 (data, targets) 가 (x, y) 로 바뀌는 것에 유의할 것!!!!!!!!###############
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            
            label_onehot = F.one_hot(y, num_classes).float() # 원핫으로 MSE loss 쓸거임
            
            #timestep = x.shape[1] # SNN은 타임스텝이 필요함 -> 포아송 인코딩은 시간축을 새로 만들기 때문에 별개로 뒀음
            
            out_fr = 0. # 출력 발화빈도를 이렇게 설정해두고, 나중에 출력인 리스트 형태로 더해진다 함
            for t in range(timestep) : 
                # timestep_data = x[:, t].unsqueeze(1)  # 각 timestep마다 (batch_size, 1) 크기로 자름
                # out_fr += model(timestep_data) # 1회 순전파
                encoded_data = encoder(x)
                out_fr += model(encoded_data)
        
        
            out_fr = out_fr / timestep
            # out_fr = torch.stack(out_fr_list).mean(dim=0)  # 타임스텝별 출력을 평균내어 합침
            
            loss = F.mse_loss(out_fr, label_onehot, reduction='none')

            weighted_loss = loss * class_weight[y].unsqueeze(1) # 가중치 곱하기 : 여긴 배치 없는데 혹시 모르니깐..?
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
        
        end_event.record()  # 종료 이벤트 기록

    torch.cuda.synchronize()  # 시간측정용 이벤트 완료 대기
    elapsed_time = start_event.elapsed_time(end_event)  # 밀리초 단위로 반환
    writer.add_scalar('elapsed_time', elapsed_time, 0) # 시간도 일단 tensorboard에 기록하기

    # 각종 평가수치들 만들고 tensorboard에 기록
    valid_loss = total_loss / len(loader)
    valid_accuracy = accuracy.compute()
    valid_f1_micro = f1_micro.compute()
    valid_f1_weighted = f1_weighted.compute()
    valid_auroc_macro = auroc_macro.compute()
    valid_auroc_weighted = auroc_weighted.compute()
    valid_auprc = auprc.compute()

    writer.add_scalar('valid_Loss', valid_loss, 0)
    writer.add_scalar('valid_Accuracy', valid_accuracy, 0)
    writer.add_scalar('valid_F1_micro', valid_f1_micro, 0)
    writer.add_scalar('valid_F1_weighted', valid_f1_weighted, 0)
    writer.add_scalar('valid_AUROC_macro', valid_auroc_macro, 0)
    writer.add_scalar('valid_AUROC_weighted', valid_auroc_weighted, 0)
    writer.add_scalar('valid_auprc', valid_auprc, 0)

    # 모델 다시 훈련으로 전환
    model.train()

    # valid loss를 반환한다. 이걸로 early stop 확인.
    return valid_loss










########### 우선순위 작업 ############

# 데이터셋의 클래스 당 비율 불일치 문제가 있으므로 가중치를 통해 균형을 맞추도록 한다.
class_weight = torch.tensor(class_weight, device=device)

# 데이터셋, 데이터로더 : test
test_dataset = MITLoader_MLP_binary(csv_file=test_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# exec_time_test : test용 exec_time 새로 만든다.
exec_time_test = saved_model_dir.split("_")[-3]

############ 1회 inference ############

# TensorBoard 폴더 설정
writer = SummaryWriter(log_dir=f"./tensorboard/{model_name}" + "_" + exec_time_test + f"_test{temp_test_fold}")

# SNN 네트워크 초기화
model = SNN_MLP(num_encoders=num_encoders, num_classes=num_classes, hidden_size=hidden_size,
                hidden_size_2=hidden_size_2, threshold_value=threshold_value, bias_option=need_bias, reset_value_residual=reset_value_residual).to(device)

encoder = encoding.PoissonEncoder() # 포아송 인코더

checkpoint = torch.load(saved_model_dir)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# 혹시 모르니 뉴런상태 초기화(가중치 초기화가 아니라 내부의 막전위 초기화임!)
functional.reset_net(model)

# valid(자체적으로 tensorboard 내장됨), 반환값으로 얼리스탑 확인하기
valid_loss = check_accuracy(test_loader, model, writer)



# 텐서보드 닫기
writer.close()

print("test finished.")