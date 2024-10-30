# 기존 CNN 필터 쪽에 IF 레이어 추가한 버전

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

# 이쪽에선 SNN 모델을 넣지 않고, 바로 jelly.layer.Linear로 바로 들어가는 것을 시도해본다. 이쪽이 오히려 학습 가능한 파라미터화 시키는 것이 아닐까? 아닌가? 해 봐야 안다.
# from temp_from_GRU import TP_encoder_MIT as TP


# import torchmetrics.functional as TF # 이걸로 메트d릭 한번에 간편하게 할 수 있다던데?

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




# 일단은 텐서보드 그대로 사용
# 텐서보드 선언(인자도 미리 뽑아두기; 나중에 json으로 바꿀 것!)
# 텐서보드 사용 유무를 json에서 설정하는 경우 눈치껏 조건문으로 비활성화!
board_class = 'binary' if num_classes == 2 else 'multi' # 클래스갯수를 1로 두진 않겠지?
writer = SummaryWriter(log_dir="./tensorboard/"+ str(model_name) + "_" + board_class
                       + "_channel" + str(encoder_filter_channel_size) + "_hidden" + str(hidden_size)
                       + "_encoderGrad" + str(encoder_requires_grad) + "_early" + str(early_stop)
                       + "_lr" + str(learning_rate) + "_threshold" + str(threshold_value)
                       + "_" + time.strftime('%Y_%m_%d_%H_%M_%S'))

# 체크포인트 위치도 상세히 갱신
checkpoint_path += str(str(model_name) + "_" + board_class
                       + "_channel" + str(encoder_filter_channel_size) + "_hidden" + str(hidden_size)
                       + "_encoderGrad" + str(encoder_requires_grad) + "_early" + str(early_stop)
                       + "_lr" + str(learning_rate) + "_threshold" + str(threshold_value)
                       + "_" + time.strftime('%Y_%m_%d_%H_%M_%S'))

# 최종에포크 저장용
lastpoint_path = checkpoint_path + "_lastEpoch.pt"

# 체크포인트 확장자 마무리
checkpoint_path += ".pt"

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
        
        # 레이어 1개로 줄이는 버전 다시 학습 필요
        # SNN 리니어 : 히든1 -> 히든2
        # self.hidden_2 = nn.Sequential(
        #     # layer.Flatten(),
        #     layer.Linear(hidden_size, hidden_size_2, bias=bias_option), # bias는 일단 기본값 True로 두기
        #     neuron.LIFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0,
        #                     v_threshold=threshold_value, tau=leak, decay_input=False),
        #     )

        # SNN 리니어 : 히든2 -> 출력
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
    
    print("validation 진행중...")

    with  torch.no_grad():
        for x, y in loader:         ############### train쪽에서 코드 복붙 시 (data, targets) 가 (x, y) 로 바뀌는 것에 유의할 것!!!!!!!!###############
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            
            label_onehot = F.one_hot(y, num_classes).float() # 원핫으로 MSE loss 쓸거임
            
            # 순전파 : SNN용으로 바꿔야 함
            timestep = x.shape[1] # SNN은 타임스텝이 필요함
            
            out_fr = 0. # 출력 발화빈도를 이렇게 설정해두고, 나중에 출력인 리스트 형태로 더해진다 함
            
            

            # # 필터연산.
            # # 전체 iter : (패딩추가된 크기 - (커널-스트라이드)) / 스트라이드.
            # # 단, 이렇게 하는 경우 매 iter마다 제로패딩을 얼만큼 넣어야 하는지도 검사해야 할 것이다.
            # # 타임스텝 앞뒤로 패딩 추가해서 계산해야 함..!
            # paddinged_size = x.shape[1] + 2 * (encoder_filter_padding)
            # paddinged_iter = (paddinged_size - (encoder_filter_kernel_size - encoder_filter_stride)) // encoder_filter_stride

            # # 데이터의 앞뒤로 제로패딩을 추가한다.
            # padded_data = F.pad(x, (encoder_filter_padding, encoder_filter_padding), "constant", 0)

            # for t in range(paddinged_iter) :  # 원래 timestep 들어가야함
            #     timestep_data = padded_data[:, t*encoder_filter_stride : t*encoder_filter_stride + encoder_filter_kernel_size]
            #     # timestep_data = data[:, t:t+encoder_filter_kernel_size]  # 각 timestep마다 (batch_size, 1) 크기로 자름, 어차피 배치 빼도 1차원 값이 되니 unsqueeze(1) 필요 없을듯? 참고로 얘도 아님
            #     out_fr += model(timestep_data) # 1회 순전파
                
                
            # 필터연산 No.2. 
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
    return valid_loss, valid_auroc_macro






########### 학습시작! ############

# raw 데이터셋 가져오기
train_dataset = MITLoader_MLP_binary(csv_file=train_path)
test_dataset = MITLoader_MLP_binary(csv_file=test_path)

# 랜덤노이즈, 랜덤쉬프트는 일단 여기에 적어두기만 하고 구현은 미뤄두자.

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # 물론 이건 그대로 써도 될 듯?
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 데이터셋의 클래스 당 비율 불일치 문제가 있으므로 가중치를 통해 균형을 맞추도록 한다.
class_weight = torch.tensor(class_weight, device=device)


# SNN 네트워크 초기화 : 이젠 상황따라 이것도 바꿔야 한다.
# model = SNN_MLP(num_encoders=num_encoders, num_classes=num_classes, hidden_size=hidden_size,
#                 hidden_size_2=hidden_size_2, kernel_size=encoder_filter_kernel_size).to(device=device) # 밑에꺼가 지금 하는 것

model = SNN_MLP(num_classes = num_classes, leak = leak_decay, hidden_size=hidden_size, hidden_size_2=hidden_size_2, 
                out_channels=encoder_filter_channel_size, kernel_size=encoder_filter_kernel_size, reset_value_residual=reset_value_residual,
                stride=encoder_filter_stride, padding=encoder_filter_padding, threshold_value=threshold_value, bias_option=need_bias).to(device=device)

# 그리고 여기에서 내부 가중치 값을 임의로 바꿀 수 있단 거겠지? : 필터연산이라 필요없음
# manual_weights = torch.linspace(encoder_min,encoder_max,steps=num_encoders).view(1,-1).to(device).transpose(1,0) # 아니 GPGPT야 이런건 어떻게 알고 찾아내주는거니
# model.encoders[0].weight = nn.Parameter(manual_weights, requires_grad=encoder_requires_grad) # 가중치 고정!
# model.encoders[0].bias.data.fill_(0.0) # bias 초기화해주는 녀석이라는데.. 일단 GPT가 제시했으니 써봄, 온전히 가중치만 보게 하니 의미있을 것 같기도 하고

# 여기서 인코더 가중치를 고정시켜야 하나??
# 모든 파라미터를 가져오되, 'requires_grad'가 False인 파라미터는 제외
train_params = [p for p in model.parameters() if p.requires_grad]

# Loss와 optimizer, scheduler (클래스별 배율 설정 포함)
pos_weight = torch.tensor([0.2, 0.8]).to(device)
criterion = nn.CrossEntropyLoss(weight=pos_weight)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(train_params, lr=learning_rate) # 가중치 고정은 제외하고 학습
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=scheduler_tmax, eta_min=scheduler_eta_min)

# inplace 오류해결을 위해 이상위치 찾기
torch.autograd.set_detect_anomaly(True)

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
        
        label_onehot = F.one_hot(targets, num_classes).float() # 원핫으로 MSE loss 쓸거임
        
        
        # 순전파 : 필터연산이므로 해당하는 만큼 자르는 스킬이 관건이 될 것
        timestep = data.shape[1] # SNN은 타임스텝이 필요함
        out_fr = 0. # 출력 발화빈도를 이렇게 설정해두고, 나중에 출력인 리스트 형태로 더해진다 함
        
        
        # 필터연산 No.2. 
        # 기존에 들어온 데이터를 바로 CNN 필터로 돌려버린다. 대신 타임스텝 반복도 모델 안쪽에서 처리한다.
        # 그렇게 나온 값을 인코더 제거된 리니어 모델에 넣는다.
        out_fr = model(data)
        
                
                
        # 여기부턴 출력값 처리에 관한 내용이니 메커니즘 건들거면 여긴 안만져도 됨
        
        # out_fr = out_fr / timestep # CNN 필터는 이걸 알아서 내줘야 함
        loss = F.mse_loss(out_fr, label_onehot, reduction='none') # 요소별로 loss를 구해야 해서 reduction을 넣는다는데..
        weighted_loss = loss * class_weight[targets].unsqueeze(1) # 가중치 곱하고 배치 차원 확장
        final_loss = weighted_loss.mean() # 요소별 뭐시기 loss를 평균내서 전체 loss 계산?
        # print(loss) # loss는 잘 나오는가? -> 아니 이거 MSE 써서 안좋아진건가?? 뭐지?


        # 얘도 일단 total_loss를 찍어봐야..겠지?
        total_loss += final_loss.item()

        # 역전파
        optimizer.zero_grad()
        # final_loss.backward(retain_graph=True) # rerain_graph로 그래프를 남기면서 역전파를 한다?? 근데 이게 없으면 역전파 2회 시도라면서 뭐시기 에러 뜨던데..
        final_loss.backward()

        # 아담 옵티마이저 : 배치 단위로 진행
        optimizer.step()


        
        # 평가지표는 전체 클래스의 발화빈도인 out_fr을 적절히 이용해서 만들기
        preds = torch.argmax(out_fr, dim=1)
        # print(preds) # 한 배치 안의 모델 예측값, 잘 나오는 것으로 보임
        accuracy.update(preds, targets)
        f1_micro.update(preds, targets)
        f1_weighted.update(preds, targets)
        auroc_macro.update(preds, targets)
        auroc_weighted.update(preds, targets)
        probabilities = F.softmax(out_fr, dim=1)[:, 1]  # 클래스 "1"의 확률 추출
        auprc.update(probabilities, targets)
        
        # SNN : 모델 초기화
        functional.reset_net(model)

        # 혹시 모르니 캐시 메모리 제거..?
        # torch.cuda.empty_cache()


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

    valid_loss, valid_auroc_macro = check_accuracy(test_loader, model) # valid(자체적으로 tensorboard 내장됨), 반환값으로 얼리스탑 확인하기

    print('epoch ' + str(epoch) + ', valid loss : ' + str(valid_loss))

    # 성능 좋게 나오면 체크포인트 저장 및 earlystop 갱신
    if early_stop_enable :
        if valid_loss < min_valid_loss : 
            min_valid_loss = valid_loss
            earlystop_counter = early_stop
            if checkpoint_save : 
                print("best performance, saving..")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,
                    }, checkpoint_path)
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
                    }, lastpoint_path)
                break # train epoch를 빠져나옴
    else : 
        final_epoch = epoch
        if epoch == num_epochs - 1 : # 얼리스탑과 별개로 최종 모델 저장
            print("last epoch model saving..")
            torch.save({
                'epoch': final_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                }, lastpoint_path)

    




print("training finished; epoch :" + str(final_epoch))

# 마지막엔 텐서보드 닫기
writer.close()