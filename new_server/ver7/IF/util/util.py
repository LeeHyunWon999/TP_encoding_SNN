import torch
from torch import nn, Tensor  # 모든 DNN 모델들
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

from executor import trainer, tester
from model import BURST

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
def get_model(args) : 
    pass

# 데이터로더 구분하여 받아오기
def get_data_loader(args) : 
    pass




# 각 모델의 순전파 동작
def propagation(model, x, args) -> float : 
    if args['model']['type'] == 'poisson' : 
        encoder = encoding.PoissonEncoder() # 포아송 인코더
        out_fr = 0.
        timestep = args['model']['args']['type_args']['timestep']

        for t in range(timestep):
            encoded_data = encoder(x)
            out_fr += model(encoded_data)

        # loss 계산 (total_loss : 1 에포크의 loss)
        out_fr /= timestep
        return out_fr
    
    elif args['model']['type'] == 'burst' : 
        burst_encoder = BURST(beta=args['model']['args']['type_args']['burst_beta'], 
                              init_th=['model']['args']['type_args']['burst_init_th']) # 버스트 인코더, 배치 안에서 소환해야 다음 배치에서 이 인코더의 남은 뉴런상태를 사용하지 않음
        out_fr = 0.
        timestep = args['model']['args']['type_args']['timestep']

        for t in range(timestep):
            encoded_data = burst_encoder(x, t)
            out_fr += model(encoded_data)

        # loss 계산 (total_loss : 1 에포크의 loss)
        out_fr /= timestep
        return out_fr

    elif args['model']['type'] == 'TP' : 
        return model(x)
    elif args['model']['type'] == 'filter_CNN' : 
        return model(x)
    else : 
        raise TypeError("Model name mismatched.")