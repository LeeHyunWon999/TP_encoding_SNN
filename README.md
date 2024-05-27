# TP_encoding_SNN

TP_encoding 방식을 SNN MLP에 적용하여 학습 잘 되는지 확인용

batch size만큼 묶어서 보내야 하므로 여기에도 GRU의 2차원 인코더를 사용하는 것이 나을 것으로 보임

`24. 05. 27. 보고
- temp_from_GRU : GRU 학습에 필요한 파일들 임시보관용, 이 안의 TP_encoder_MIT.py에 인코딩용 뉴런이 정의되어 있음
- 이걸 MIT_SNN_main.py의 71번 줄, SNN_binary 모델 안에 집어넣으려 시도중(76째 줄)
- 생각중인 방법 : nn.ModuleList 형식으로 받아다가 for문으로 넣기
- 대안 : jelly.layer.Linear(1,인코더뉴런갯수) 로 두기?