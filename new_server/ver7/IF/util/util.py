import torch
from torch import nn, Tensor  # 모든 DNN 모델들
from torch import optim  # SGD, Adam 등
from torch.optim.lr_scheduler import CosineAnnealingLR # 코사인스케줄러(옵티마이저 보조용)
import torch.nn.functional as F  # 일부 활성화 함수 등 파라미터 없는 함수에 사용
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

from executor import trainer, tester
from data import data_loader
import model.model as model
from model.model import BURST

def execute(args) :     
    if args['executor']['type'] == 'trainer' : 
        print('훈련모드 진입')
        trainer.trainer(args).train()
    elif args['executor']['type'] == 'tester' : 
        print('테스트모드 진입')
        tester.tester(args).test()
    else : 
        print('오류. 인자를 다시 확인하십시오.')


# 모델 구분하여 받아오기
def get_model(args, device) : 
    if args['type'] == 'poisson' : 
        return model.SNN_MLP(num_encoders=args['args']['input_size'], num_classes=args['args']['num_classes'], 
                             hidden_size=args['args']['hidden_size'], hidden_size_2=None, 
                             threshold_value=args['args']['threshold'], bias_option=args['args']['need_bias'], 
                             reset_value_residual=args['args']['reset_value_residual'])
    elif args['type'] == 'burst' : # 일단은 poisson과 동일한 구성
        return model.SNN_MLP(num_encoders=args['args']['input_size'], num_classes=args['args']['num_classes'], 
                             hidden_size=args['args']['hidden_size'], hidden_size_2=None, 
                             threshold_value=args['args']['threshold'], bias_option=args['args']['need_bias'], 
                             reset_value_residual=args['args']['reset_value_residual'])
    elif args['type'] == 'TP' : 
        return model.TP(num_classes = args['args']['num_classes'], 
                        hidden_size=args['args']['hidden_size'], hidden_size_2=args['args']['hidden_size_2'], 
                        threshold_value=args['args']['threshold'], bias_option=args['args']['need_bias'], 
                        reset_value_residual=args['args']['reset_value_residual'], 
                        encoder_min = args['args']['type_args']['encoder_min'], encoder_max = args['args']['type_args']['encoder_max'], device = device)
    elif args['type'] == 'filter_CNN' : 
        return model.filter_CNN(num_classes = args['args']['num_classes'], hidden_size=args['args']['hidden_size'], hidden_size_2=None, 
                                out_channels=args['args']['type_args']['channel'], kernel_size=args['args']['type_args']['window'], 
                                stride=args['args']['type_args']['stride'], padding=args['args']['type_args']['padding'], 
                                threshold_value=args['args']['threshold'], bias_option=args['args']['need_bias'], 
                                reset_value_residual=args['args']['reset_value_residual'])
    elif args['type'] == 'TP_2D' : 
        return model.TP_2D(num_classes = args['args']['num_classes'], input_channel=args['args']['input_channel'],
                        hidden_size=args['args']['hidden_size'], hidden_size_2=args['args']['hidden_size_2'], 
                        threshold_value=args['args']['threshold'], bias_option=args['args']['need_bias'], 
                        reset_value_residual=args['args']['reset_value_residual'], 
                        encoder_min = args['args']['type_args']['encoder_min'], encoder_max = args['args']['type_args']['encoder_max'], device = device)
    elif args['type'] == 'filter_CNN_2D' : 
        return model.filter_CNN_2D(num_classes = args['args']['num_classes'],  input_channel=args['args']['input_channel'], hidden_size=args['args']['hidden_size'], hidden_size_2=None, 
                                out_channels=args['args']['type_args']['channel'], kernel_size=args['args']['type_args']['window'], 
                                stride=args['args']['type_args']['stride'], padding=args['args']['type_args']['padding'], 
                                threshold_value=args['args']['threshold'], bias_option=args['args']['need_bias'], 
                                reset_value_residual=args['args']['reset_value_residual'])
    

# 데이터로더 구분하여 받아오기 : trainer
def get_data_loader_train(args) : 
    if args['type'] == 'CinC' : 
        return data_loader.CinC_Loader(ts_file_path=args['args']['train_path'])
    elif args['type'] == 'MIT-BIH' : 
        return data_loader.MITLoader_MLP_binary(csv_file=args['args']['train_path'])
    elif args['type'] == 'CinC_original' : 
        return data_loader.CinC_original_Loader(npy_dir=args['args']['train_path'], label_json_path=args['args']['train_label_path'])
    else : 
        raise TypeError("지원되지 않는 데이터로더 인자입니다.")

# 데이터로더 구분하여 받아오기 : tester
def get_data_loader_test(args) : 
    if args['type'] == 'CinC' : 
        return data_loader.CinC_Loader(ts_file_path=args['args']['test_path'])
    elif args['type'] == 'MIT-BIH' : 
        return data_loader.MITLoader_MLP_binary(csv_file=args['args']['test_path'])
    else : 
        raise TypeError("지원되지 않는 데이터로더 인자입니다.")

# 옵티마이저 겟
def get_optimizer(train_params, args) : 
    if args['type'] == 'Adam':
        return optim.Adam(train_params, lr=args['lr'])
    else : 
        raise TypeError("지원되지 않는 옵티마이저 인자입니다.")

# 스케줄러 겟
def get_scheduler(optimizer, args) : 
    if args['type'] == 'CosineAnnealingLR' : 
        return CosineAnnealingLR(optimizer=optimizer, T_max=args['args']['T_max'], eta_min=args['args']['eta_min'])
    else : 
        raise TypeError("지원되지 않는 스케줄러 인자입니다.")



# 각 모델의 순전파 동작 (forward()에 합치는 경우 동일한 분류모델을 사용하는 poisson, burst 간의 구분이 forward() 안에서 이뤄져야 하므로 편의상 분리)
# TP, filterCNN의 경우도 분리할 수 있으나.. 기존 코드가 인코더-모델 일체형이라 일단 보류
def propagation(model, x, args, args_data_loader) -> float : 
    if args['type'] == 'poisson' : 
        encoder = encoding.PoissonEncoder() # 포아송 인코더
        out_fr = 0.
        timestep = args['args']['type_args']['timestep']

        for t in range(timestep):
            encoded_data = encoder(x)
            out_fr += model(encoded_data)

        # loss 계산 (total_loss : 1 에포크의 loss)
        out_fr /= timestep
        return out_fr
    
    elif args['type'] == 'burst' : 
        burst_encoder = BURST(beta=args['args']['type_args']['burst_beta'], 
                              init_th=args['args']['type_args']['burst_init_th']) # 버스트 인코더, 배치 안에서 소환해야 다음 배치에서 이 인코더의 남은 뉴런상태를 사용하지 않음
        out_fr = 0.
        timestep = args['args']['type_args']['timestep']

        for t in range(timestep):
            encoded_data = burst_encoder(x, t)
            out_fr += model(encoded_data)

        # loss 계산 (total_loss : 1 에포크의 loss)
        out_fr /= timestep
        return out_fr

    elif args['type'] == 'TP' : 
        return model(x)
    elif args['type'] == 'filter_CNN' : 
        return model(x)
    elif args['type'] == 'TP_2D' : 
        if args_data_loader['type'] == 'CinC' : 
            assert x.shape[-1] == 24705, "Input feature size should be 61 x 405 = 24705"
            x = x.view(-1, 61, 405)
            return model(x)
        elif args_data_loader['type'] == 'CinC_original' : 
            assert x.shape[-1] == 3000, "Input feature size should be 6 x 500 = 3000"
            x = x.view(-1, 6, 500)
            return model(x)
        else : 
            raise TypeError("TP_2D 모델의 해당 데이터로더에 대한 데이터 후처리 방식이 지정되지 않았습니다.")
    elif args['type'] == 'filter_CNN_2D' : 
        if args_data_loader['type'] == 'CinC' : 
            assert x.shape[-1] == 24705, "Input feature size should be 61 x 405 = 24705"
            x = x.view(-1, 61, 405)
            return model(x)
        elif args_data_loader['type'] == 'CinC_original' : 
            assert x.shape[-1] == 3000, "Input feature size should be 6 x 500 = 3000"
            x = x.view(-1, 6, 500)
            return model(x)
        else : 
            raise TypeError("filter_CNN_2D 모델의 해당 데이터로더에 대한 데이터 후처리 방식이 지정되지 않았습니다.")
    else : 
        raise TypeError("지원되지 않는 순전파 인자입니다 : 모델명 불일치.")
    


# loss 구분 (쓸 일은 없겠지만.. 나중에 loss 변경 작업 시 필요할 수 있으므로 일단 넣어두기)
def get_loss(out_fr, label_onehot, args) : 
    if args['type'] == 'MSE_loss' : 
        return F.mse_loss(out_fr, label_onehot, reduction='none')
    else : 
        raise TypeError("지원되지 않는 손실함수 인자입니다.")