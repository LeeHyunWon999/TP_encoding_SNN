tester 작성 계획
- 일단 .pt파일을 직접 가져올 것 : fold 갯수만큼 필요하되, 하나의 config에 합치기
- config파일은 train과 거의 같지만 type이 tester인 것, 그리고 checkpoint의 path가 fold 갯수만큼 리스트로 존재한다 정도만 차이로 두면 될 것으로 보임
- tester.py에선 모델 선언 후 체크포인트 로드하고, 이후 데이터로더 등 필요한 녀석들 불러온 뒤에 validation 메소드 그대로 적용하면 될 것으로 보임