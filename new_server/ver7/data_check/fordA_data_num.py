from collections import Counter

def analyze_ford_a_ts(ts_file_path):
    with open(ts_file_path, 'r') as f:
        lines = f.readlines()

    # @data 이후 유효 라인 찾기
    data_start_idx = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
    data_lines = [line.strip() for line in lines[data_start_idx:] if line.strip()]

    label_counter = Counter()
    data_lengths = []

    for line in data_lines:
        if ':' not in line:
            continue

        signal_str, label_str = line.rsplit(':', 1)
        signal_values = signal_str.split(',')
        label = int(label_str.strip())

        label_counter[label] += 1
        data_lengths.append(len(signal_values))

    # 고유 시계열 길이 확인
    unique_lengths = set(data_lengths)

    print("📊 FordA 데이터셋 요약:")
    print(f"· 총 데이터 수           : {len(data_lines)}개")
    print(f"· 각 라벨 개수           : {dict(label_counter)}")
    print(f"· 데이터포인트 길이 종류 : {unique_lengths}")
    if len(unique_lengths) == 1:
        print(f"✅ 모든 시계열 길이가 동일합니다. 길이 = {unique_lengths.pop()}")
    else:
        print("❗ 데이터포인트 길이가 서로 다릅니다.")

# 사용
analyze_ford_a_ts("/home/hschoi/data/leehyunwon/time_series_FordA/FordA_TRAIN.ts")
