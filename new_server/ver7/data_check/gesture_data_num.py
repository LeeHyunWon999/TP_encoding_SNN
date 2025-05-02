def analyze_ts_file(ts_file_path):
    with open(ts_file_path, 'r') as f:
        lines = f.readlines()

    # 메타데이터 추출
    series_length = None
    for line in lines:
        if line.lower().startswith("@serieslength"):
            series_length = int(line.strip().split()[-1])
            break

    # @data 이후부터 실제 데이터 시작
    data_start_idx = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
    data_lines = [line for line in lines[data_start_idx:] if line.strip()]

    # 첫 번째 데이터포인트 길이 측정
    first_line_values = data_lines[0].strip().split(',')
    total_length = len(first_line_values)

    # 차원 수 추정
    if series_length:
        num_dimensions = total_length // series_length
    else:
        num_dimensions = 1  # fallback
        series_length = total_length  # fallback

    print("📊 .ts 파일 정보 요약")
    print(f"· 총 데이터 수        : {len(data_lines)}개")
    print(f"· 하나의 데이터 길이  : {total_length} (값 개수)")
    print(f"· 시계열 길이         : {series_length}")
    print(f"· 추정 차원 수        : {num_dimensions}")

# 사용 예시
ts_path = "/home/hschoi/data/leehyunwon/time_series_Gesture/UWaveGestureLibraryAll_TRAIN.ts"
analyze_ts_file(ts_path)
