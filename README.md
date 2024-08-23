# TP_encoding_SNN

TP_encoding 방식을 SNN MLP에 적용하여 학습 잘 되는지 확인용

batch size만큼 묶어서 보내야 하므로 여기에도 GRU의 2차원 인코더를 사용하는 것이 나을 것으로 보임

- ver3 : 학습에 쓰이는 파라미터 등 추가, 얼리스탑은 valid_loss와 valid_auroc가 갱신되는 경우 초기화
 
`24. 07. 20. 특이사항 : 현재 사용중인 모델들
- ver2_poisson_2 : 1024 2층 쌓은 포아송
- ver3 : 추가 하이퍼파라미터 넣은 버전(단순 iter은 성능 좋지 않았으므로 filter 위주로 볼 것)
- ver3_filter_CNN : 필터계열 중 가장 성능 좋음


`24. 08. 22. 버전4 업데이트 시작
- 이전 버전들에서 계속 사용되는 모델들만 추려서 버전통일 : ver2_poisson_2(넘버링 제거), ver2_burst, ver3_filter_CNN_IF
- 랜덤시드 고정으로 성능 비교 일관성 향상
- 최고성능 나왔을 때 별도 폴더에 체크포인트 저장하는 기능 추가 : 근데 이제 옵션인
- Neu+ inference 시 뉴런의 초기화 방법이 잔차가 아닌 0이므로 해당 기능 변경
- threshold 파라미터 추가 : 가중치를 92로 지정하는 경우 학습되는 파라미터들이 전반적으로 높게 형성되는지에 대한 확인 필요(CNN 모델도 일관성을 위해 통일하는 편이 나아보임)
- 최종 추가 파라미터 : random_seed, checkpoint_path, threshold

### 나중에 할 일
- 리듬 데이터셋 입수하는 경우 그쪽에선 cross-validation도 추가할 것