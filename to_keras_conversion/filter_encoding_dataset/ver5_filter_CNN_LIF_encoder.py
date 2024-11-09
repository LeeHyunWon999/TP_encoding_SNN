# ver5_filter_CNN_LIF 버전으로 학습한 모델에서 필터 인코더와 IF레이어까지를 거친 activation 값만을 인코딩 데이터로 뽑아내기
# 학습 시에 쓰던 config 파일을 그대로 가져올 순 없으므로, 파라미터는 직접 보고 옮길 것

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



# Cuda 써야겠지?
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # GPU 번호별로 0번부터 나열
os.environ["CUDA_VISIBLE_DEVICES"]= "0"   # 이쪽 서버는 GPU 4개임
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
model_name = json_data['model_name']
num_classes = json_data['num_classes']
num_encoders = json_data['num_encoders']
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
encoder_type = json_data['encoder_type']
encoder_tp_iter_repeat = json_data['encoder_tp_iter_repeat']
encoder_filter_kernel_size = json_data['encoder_filter_kernel_size']
encoder_filter_stride = json_data['encoder_filter_stride']
encoder_filter_padding = json_data['encoder_filter_padding']
encoder_filter_channel_size = json_data['encoder_filter_channel_size'] # CNN 스타일로 가려면 채널갯수로 깊게 분석해야 할 것이다.
random_seed = json_data['random_seed']
checkpoint_save = json_data['checkpoint_save']
checkpoint_path = json_data['checkpoint_path']
threshold_value = json_data['threshold_value']
reset_value_residual = json_data['reset_value_residual']
leak_decay = json_data['leak_decay']
need_bias = json_data['need_bias']
saved_model_path = json_data['saved_model_path']
outputPath = json_data['outputPath']


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





# 참고 : 이것 외에도 에포크, Loss까지 찍어야 하니 참고할 것!
earlystop_counter = early_stop
min_valid_loss = float('inf')
max_valid_auroc_macro = -float('inf')
final_epoch = 0 # 마지막에 최종 에포크 확인용


# 이제 복잡한 인코더들을 따로 여기에 정의해야 한다. 근데 이제 그냥 반복하는건 for문으로 꼬라박으면 되니까 필터연산하는 녀석만 있으면 될 듯?



# 여기선 CNN 인코딩 방식을 취했다.
class SNN_MLP(nn.Module):
    def __init__(self, num_classes, leak, hidden_size, hidden_size_2, out_channels, kernel_size, stride, padding, threshold_value, bias_option, reset_value_residual):
        super().__init__()
        
        # CNN 인코더 필터 : 이건 그냥 갈긴다.
        self.cnn_encoders = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias_option) # 여기도 bias가 있다 함
        
        # CNN 인코더 IF뉴런 : 이거 추가해서 인코더 완성하기
        self.cnn_IF_layer = neuron.LIFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0,
                                            v_threshold=threshold_value, tau=leak, decay_input=False)
        
        # SNN 리니어 : 인코더 입력 -> 히든1
        self.hidden = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(out_channels, hidden_size, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.LIFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0,
                            v_threshold=threshold_value, tau=leak, decay_input=False),
            )
        
        # 레이어 1개로 줄이는 버전 다시 학습 필요 (살려둬서 가중치가 저장되긴 했지만 학습되지 않은 녀석이므로 제외시킬 것!)
        # SNN 리니어 : 히든1 -> 히든2
        self.hidden_2 = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size, hidden_size_2, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.LIFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0,
                            v_threshold=threshold_value, tau=leak, decay_input=False),
            )

        # SNN 리니어 : 히든 -> 출력
        self.layer = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size, num_classes, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.LIFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0,
                            v_threshold=threshold_value, tau=leak, decay_input=False),
            )

    def forward(self, x: torch.Tensor, encoding_save: bool = False):
        results = 0. # for문이 모델 안에 있으므로 밖에다가는 이녀석을 내보내야 함
        
        # CNN 필터는 채널 차원이 추가되므로 1번 쪽에 채널 차원 추가
        x = x.unsqueeze(1)
        # CNN 필터 통과시키기
        x = self.cnn_encoders(x)
        # print(x.shape)
        timestep_size = x.shape[2]
        # 근데 이제 이렇게 바꾼 데이터는 (배치, 채널, 출력크기) 만큼의 값을 갖고 있으니 여기서 나온 값들을 하나씩 잘라서 다음 레이어로 넘겨야 한다.
        for i in range(timestep_size) : 
            x_slice = x[:,:,i].squeeze() # 이러면 출력크기 차원이 사라지고 (배치, 채널)만 남겠지?
            x_slice = self.cnn_IF_layer(x_slice) # CNN 필터 이후 IF 레이어 거치기
            
            x_slice = self.hidden(x_slice)
            # x_slice = self.hidden_2(x_slice) # 3레이어로 변경 : Neu+의 4레이어가 동작하지 않음
            x_slice = self.layer(x_slice)
            results += x_slice  # 결과를 리스트에 저장(출력발화값은 전부 더하는 식으로)
        
        return results / timestep_size
    
    




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


# test 데이터로 정확도 측정, 여기선 특별히 필터 인코더 + IF레이어까지만 동작 후 인코딩 데이터 반환
def check_accuracy(loader, model):

    # 각종 메트릭들 리셋(train에서 에폭마다 돌리므로 얘도 에폭마다 들어감)
    total_loss = 0
    
    # 모델 상태 불러오기 (model_state_dict만 가져옴)
    checkpoint = torch.load('/home/hschoi/data/leehyunwon/ECG-SNN/SNN_MLP_ver5_filter_CNN_LIF_3layer_binary_channel1000_hidden1000_encoderGradTrue_early25_lr0.001_threshold1.0_2024_10_29_11_14_10_lastEpoch.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 모델 평가용으로 전환
    model.eval()
    
    print("validation 진행중...")

    with  torch.no_grad():
        for x, y in loader:         ############### train쪽에서 코드 복붙 시 (data, targets) 가 (x, y) 로 바뀌는 것에 유의할 것!!!!!!!!###############
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            
            label_onehot = F.one_hot(y, num_classes).float() # 원핫으로 MSE loss 쓸거임
            
            out_fr = 0. # 출력 발화빈도를 이렇게 설정해두고, 나중에 출력인 리스트 형태로 더해진다 함
            
            
                
            # 필터연산 No.2. 
            out_fr = model(x) # 앞으로도 그렇겠지만, 순전파꺼 넣는다고 x 말고 data 넣는 치명적 실수 하지 말 것 !!!
        

            
            loss = F.mse_loss(out_fr, label_onehot, reduction='none')

            weighted_loss = loss * class_weight[y].unsqueeze(1) # 가중치 곱하기
            final_loss = weighted_loss.mean() # 요소별 뭐시기 loss를 평균내서 전체 loss 계산?
            
            # 여기에도 total loss 찍기
            total_loss += final_loss.item()

            
            # 얘도 SNN 모델이니 초기화 필요
            functional.reset_net(model)


    # 각종 평가수치들 만들고 tensorboard에 기록
    valid_loss = total_loss / len(loader)

    # valid loss를 반환한다. 이걸로 early stop 확인.
    print("valid loss :", valid_loss)






########### 학습시작! ############

# raw 데이터셋 가져오기
train_dataset = MITLoader_MLP_binary(csv_file=train_path)
test_dataset = MITLoader_MLP_binary(csv_file=test_path)

# 랜덤노이즈, 랜덤쉬프트는 일단 여기에 적어두기만 하고 구현은 미뤄두자.

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # 물론 이건 그대로 써도 될 듯?
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 데이터셋의 클래스 당 비율 불일치 문제가 있으므로 가중치를 통해 균형을 맞추도록 한다.
class_weight = torch.tensor(class_weight, device=device)


# SNN 네트워크 초기화
model = SNN_MLP(num_classes = num_classes, leak = leak_decay, hidden_size=hidden_size, hidden_size_2=hidden_size_2, 
                out_channels=encoder_filter_channel_size, kernel_size=encoder_filter_kernel_size, reset_value_residual=reset_value_residual,
                stride=encoder_filter_stride, padding=encoder_filter_padding, threshold_value=threshold_value, bias_option=need_bias).to(device=device)

# 여기서 인코더 가중치를 고정시켜야 하나??
# 모든 파라미터를 가져오되, 'requires_grad'가 False인 파라미터는 제외
train_params = [p for p in model.parameters() if p.requires_grad]

# Loss와 optimizer, scheduler (클래스별 배율 설정 포함)
pos_weight = torch.tensor([0.2, 0.8]).to(device)
criterion = nn.CrossEntropyLoss(weight=pos_weight)
optimizer = optim.Adam(train_params, lr=learning_rate) # 가중치 고정은 제외하고 학습
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=scheduler_tmax, eta_min=scheduler_eta_min)

# inplace 오류해결을 위해 이상위치 찾기
torch.autograd.set_detect_anomaly(True)

# 들여온 모델 validation 진행하여 잘 불러와졌는지 확인
# check_accuracy(test_loader, model)




# 이제 필터 인코딩을 진행한다.

# 모델 상태 불러오기 (model_state_dict만 가져옴)
checkpoint = torch.load(saved_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

encoded_data = []

with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device).squeeze(1)
        
        activation_spikes = []  # IF 레이어를 거친 activation spike 값을 저장할 리스트
        
        # 이 밑의 동작은 모델 클래스의 forward 초반 구간을 그대로 가져와야 한다.
        
        # CNN 필터 통과시키기
        x = data.unsqueeze(1)  # 채널 차원 추가
        x = model.cnn_encoders(x)

        # 첫 번째 IF 레이어까지만 통과시키기
        timestep_size = x.shape[2]
        for i in range(timestep_size):
            x_slice = x[:, :, i].squeeze()
            activation_spike = model.cnn_IF_layer(x_slice)  # 첫 번째 IF 레이어를 거친 값
            activation_spikes.append(activation_spike)
            print(np.shape(activation_spikes[-1].cpu()))  # 가장 최근의 activation_spike의 shape 출력
            print(len(activation_spikes))

        # activation_spikes를 (batch_size, timestep_size, channel) 형태로 쌓기
        activation_spikes = torch.stack(activation_spikes, dim=1)  # timestep_size 차원에서 쌓기
        encoded_data.append(activation_spikes)  # encoded_data에 추가
        
        # SNN 모델에서 이건 필수다.
        functional.reset_net(model)

# 모든 배치의 encoded_data를 하나의 텐서로 결합하여 (총 데이터 개수, timestep_size, channel) 형태로 만듦
encoded_data = torch.cat(encoded_data, dim=0).cpu()

print(encoded_data.shape)  # 최종 형태 출력

# 출력 시의 형태는 [데이터갯수, 타임스텝, 채널(=뉴런갯수)] 일 것이다. 이를 다른 데이터셋과 동일하게 [데이터갯수, 뉴런, 타임스텝] 형태로 바꿔야 한다.
encoded_data = encoded_data.permute(0, 2, 1)
print(encoded_data.shape)


# 이제 이걸 저장하기만 하면 된다..
np.save(json_data["outputPath"] + 'mitbih_test_' + str(encoded_data.shape[1]) + '_channel_'
        + str(encoded_data.shape[2]) + '_timestep_filterCNN.npy', encoded_data) # 일단 이거 되긴 하는지 확인 필요



print("저장 완료")