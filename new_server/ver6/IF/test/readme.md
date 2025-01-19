사용법

- 각 모델의 config.json 안에 saved_model_dir, temp_test_fold 이 들어가 있다. 각각 저장된 모델과 결과 파일명에 들어갈 fold명이다.
- 설정값도 저장된 config에서 바로 적용하면 좋겠지만, 일단은 수작업으로 둔다. 필요시 이를 반영하는 코드 작성할 것.
- GPU, 저장파일명, 앞서 설명한 2개의 변수 정도, 그 외 설정을 변경한 뒤 fold 하나씩 inference하는 동작이다.