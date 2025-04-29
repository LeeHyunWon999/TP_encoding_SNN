# Imports
import os
import torch
from torch import nn, Tensor  # 모든 DNN 모델들

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer


############################################ poisson & burst 방식(인코더만 다르고 모델구조는 동일), 1D, 2D 범용 (2D도 flatten 상태로 수행) ############################################
class SNN_MLP(nn.Module):
    def __init__(self, num_classes, num_encoders, hidden_size, hidden_size_2, threshold_value, bias_option, reset_value_residual):
        super().__init__()
        
        # SNN 리니어 : 인코더 입력 -> 히든
        self.hidden = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(num_encoders, hidden_size, bias = bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )
        
        # 레이어 하나만으로 시도해보자.
        # # SNN 리니어 : 인코더 히든 -> 히든2
        # self.hidden2 = nn.Sequential(
        #     # layer.Flatten(),
        #     layer.Linear(hidden_size, hidden_size_2, bias = bias_option), # bias는 일단 기본값 True로 두기
        #     neuron.IFNode(surrogate_function=surrogate.ATan(), v_reset=0.0, v_threshold=threshold_value),
        #     )

        # SNN 리니어 : 히든2 -> 출력
        self.layer = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size, num_classes, bias = bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )
        
    
    # 여기서 인코딩 레이어만 딱 빼면 된다.
    def forward(self, x: torch.Tensor):
        x = self.hidden(x)
        # x = self.hidden2(x)
        return self.layer(x)
    

# burst 인코더
class BURST(nn.Module):
    def __init__(self, beta=2, init_th=0.0625, device='cuda') -> None:
        super().__init__()
        
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


############################################ TP 방식, 1D ############################################
class TP(nn.Module):
    def __init__(self, num_classes, hidden_size, hidden_size_2, threshold_value, bias_option, reset_value_residual, encoder_min, encoder_max, device, encoder_learnable):
        super().__init__()
        
        # SNN 인코더 : 채널 크기만큼 확장하기
        self.encoder = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(1, hidden_size, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )

        # SNN 리니어 : 인코더 출력 -> 히든
        self.hidden = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size, hidden_size_2, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )
        

        # SNN 리니어 : 히든 -> 출력
        self.layer = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size_2, num_classes, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )


        # 인코더 가중치 수동지정
        manual_weights = torch.linspace(encoder_min,encoder_max,steps=hidden_size).view(1,-1).to(device).transpose(1,0) # 0.2부터 2.0까지 인코더 뉴런 수만큼 지정
        self.encoder[0].weight = nn.Parameter(manual_weights) # 대입
        self.encoder[0].bias.data.fill_(0.0) # bias도 0으로 초기화

        # 인코더 가중치 학습여부는 TP냐 TP_learnable이냐 따라 다름!
        for param in self.encoder.parameters():
            param.requires_grad = encoder_learnable


    def forward(self, x: torch.Tensor):
        results = 0. # for문이 모델 안에 있으므로 밖에다가는 이녀석을 내보내야 함
        # print(x.shape) # (배치크기, 187) 모양임
        
        timestep_size = x.shape[1] # 187 timestep을 만들어야 함
        # 근데 이제 이렇게 바꾼 데이터는 (배치, 출력크기) 만큼의 값을 갖고 있으니 여기서 나온 값들을 하나씩 잘라서 다음 레이어로 넘겨야 한다.
        for i in range(timestep_size) : 
            x_slice = x[:,i].squeeze().unsqueeze(1) # 슬라이스 진행 후 256, 1 크기가 되도록 shape 수정
            x_slice = self.encoder(x_slice)
            x_slice = self.hidden(x_slice)
            x_slice = self.layer(x_slice)
            results += x_slice  # 결과를 리스트에 저장(출력발화값은 전부 더하는 식으로)
        # results = torch.stack(results, dim=0) # 텐서로 바꾸기
        return results / timestep_size
    


    ############################################ TP 방식, 2D ############################################
class TP_2D(nn.Module):
    def __init__(self, num_classes, input_channel, hidden_size, hidden_size_2, threshold_value, bias_option, reset_value_residual, encoder_min, encoder_max, device, encoder_learnable):
        super().__init__()
        
        # SNN 인코더 : 채널 크기만큼 확장하기
        self.encoder = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(input_channel, hidden_size, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )

        # SNN 리니어 : 인코더 출력 -> 히든
        self.hidden = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size, hidden_size_2, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )
        

        # SNN 리니어 : 히든 -> 출력
        self.layer = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size_2, num_classes, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )


        # 인코더 가중치 수동지정
        manual_weights = torch.linspace(encoder_min,encoder_max,steps=hidden_size).view(1,-1).to(device).transpose(1,0) # 0.2부터 2.0까지 인코더 뉴런 수만큼 지정
        manual_weights = manual_weights.expand(-1, input_channel).clone()  # shape 확장 필요 : (hidden_size, 채널널) 크기가 되도록 지정
        self.encoder[0].weight = nn.Parameter(manual_weights) # 대입
        self.encoder[0].bias.data.fill_(0.0) # bias도 0으로 초기화

        # 인코더 가중치 학습여부는 TP냐 TP_learnable이냐 따라 다름!
        for param in self.encoder.parameters():
            param.requires_grad = encoder_learnable


    def forward(self, x: torch.Tensor):
        results = 0. # for문이 모델 안에 있으므로 밖에다가는 이녀석을 내보내야 함
        # print(x.shape) # (배치크기, 채널(61, 6 등), 시간축(405, 500 등등) 모양이 될 것
        
        timestep_size = x.shape[2] # 405 timestep을 만들어야 함
        # 근데 이제 이렇게 바꾼 데이터는 (배치, 출력크기) 만큼의 값을 갖고 있으니 여기서 나온 값들을 하나씩 잘라서 다음 레이어로 넘겨야 한다.
        for i in range(timestep_size) : 
            x_slice = x[:,:,i] # 슬라이스 진행 후 256, 61 등의 크기가 되도록 shape 수정(squeeze 등은 여기서 필요없음)
            x_slice = self.encoder(x_slice)
            x_slice = self.hidden(x_slice)
            x_slice = self.layer(x_slice)
            results += x_slice  # 결과를 리스트에 저장(출력발화값은 전부 더하는 식으로)
        # results = torch.stack(results, dim=0) # 텐서로 바꾸기
        return results / timestep_size


    



############################################ FTP 방식, 1D ############################################
class filter_CNN(nn.Module):
    def __init__(self, num_classes, hidden_size, hidden_size_2, out_channels, kernel_size, stride, padding, threshold_value, bias_option, reset_value_residual):
        super().__init__()
        
        # CNN 인코더 필터 : 이건 그냥 갈긴다.
        self.cnn_encoders = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias_option) # 여기도 bias가 있다 함
        
        # CNN 인코더 IF뉴런 : 이거 추가해서 인코더 완성하기
        self.cnn_IF_layer = neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value)
        
        # SNN 리니어 : 인코더 입력 -> 히든
        self.hidden = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(out_channels, hidden_size, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )
        
        # 레이어 1개로 줄이는 버전 다시 학습 필요
        # SNN 리니어 : 히든1 -> 히든2
        # self.hidden_2 = nn.Sequential(
        #     # layer.Flatten(),
        #     layer.Linear(hidden_size, hidden_size_2, bias=bias_option), # bias는 일단 기본값 True로 두기
        #     neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
        #     )

        # SNN 리니어 : 히든 -> 출력
        self.layer = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size, num_classes, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )

    def forward(self, x: torch.Tensor):
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
            # x_slice = self.hidden_2(x_slice)
            x_slice = self.layer(x_slice)
            results += x_slice  # 결과를 리스트에 저장(출력발화값은 전부 더하는 식으로)
        # results = torch.stack(results, dim=0) # 텐서로 바꾸기
        return results / timestep_size
    



############################################ FTP 방식, 2D ############################################
class filter_CNN_2D(nn.Module):
    def __init__(self, num_classes, input_channel, hidden_size, hidden_size_2, out_channels, kernel_size, stride, padding, threshold_value, bias_option, reset_value_residual):
        super().__init__()
        
        # CNN 인코더 필터 : 이건 그냥 갈긴다.
        self.cnn_encoders = nn.Conv1d(in_channels=input_channel, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias_option) # 여기도 bias가 있다 함
        
        # CNN 인코더 IF뉴런 : 이거 추가해서 인코더 완성하기
        self.cnn_IF_layer = neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value)
        
        # SNN 리니어 : 인코더 입력 -> 히든
        self.hidden = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(out_channels, hidden_size, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )
        
        # 레이어 1개로 줄이는 버전 다시 학습 필요
        # SNN 리니어 : 히든1 -> 히든2
        # self.hidden_2 = nn.Sequential(
        #     # layer.Flatten(),
        #     layer.Linear(hidden_size, hidden_size_2, bias=bias_option), # bias는 일단 기본값 True로 두기
        #     neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
        #     )

        # SNN 리니어 : 히든 -> 출력
        self.layer = nn.Sequential(
            # layer.Flatten(),
            layer.Linear(hidden_size, num_classes, bias=bias_option), # bias는 일단 기본값 True로 두기
            neuron.IFNode(surrogate_function=surrogate.ATan(),v_reset= None if reset_value_residual else 0.0, v_threshold=threshold_value),
            )

    def forward(self, x: torch.Tensor):
        results = 0. # for문이 모델 안에 있으므로 밖에다가는 이녀석을 내보내야 함
        
        # CNN 필터는 채널 차원이 추가되므로 1번 쪽에 채널 차원 추가 : 2D 방식에선 원래부터 2차원으로 입력되므로 필요없음
        #x = x.unsqueeze(1)

        # CNN 필터 통과시키기
        x = self.cnn_encoders(x) # [배치, 출력채널(기본 1024), 새로운 timestep] 형태가 될 것임
        # print(x.shape)
        timestep_size = x.shape[2]
        # 근데 이제 이렇게 바꾼 데이터는 (배치, 채널, 출력크기) 만큼의 값을 갖고 있으니 여기서 나온 값들을 하나씩 잘라서 다음 레이어로 넘겨야 한다.
        for i in range(timestep_size) : 
            x_slice = x[:,:,i].squeeze() # 이러면 출력크기 차원이 사라지고 (배치, 채널)만 남겠지?
            x_slice = self.cnn_IF_layer(x_slice) # CNN 필터 이후 IF 레이어 거치기
            x_slice = self.hidden(x_slice)
            # x_slice = self.hidden_2(x_slice)
            x_slice = self.layer(x_slice)
            results += x_slice  # 결과를 리스트에 저장(출력발화값은 전부 더하는 식으로)
        # results = torch.stack(results, dim=0) # 텐서로 바꾸기
        return results / timestep_size