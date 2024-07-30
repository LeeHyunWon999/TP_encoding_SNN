# TP_encoding_SNN

TP_encoding 방식을 SNN MLP에 적용하여 학습 잘 되는지 확인용

batch size만큼 묶어서 보내야 하므로 여기에도 GRU의 2차원 인코더를 사용하는 것이 나을 것으로 보임
 
`24. 07. 20. 특이사항 : 현재 사용중인 모델들
- ver2_poisoon_2 : 1024 2층 쌓은 포아송
- ver3 : 추가 하이퍼파라미터 넣은 버전(단순 iter은 성능 좋지 않았으므로 filter 위주로 볼 것)
- ver3_filter_CNN : 필터계열 중 가장 성능 좋음