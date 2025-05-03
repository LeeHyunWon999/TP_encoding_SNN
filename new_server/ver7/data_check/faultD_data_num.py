from collections import Counter

def analyze_ts_multiclass(ts_file_path):
    with open(ts_file_path, 'r') as f:
        lines = f.readlines()

    # @data 이후부터 유효 데이터
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

    unique_lengths = set(data_lengths)
    total = len(data_lines)

    print("📊 FaultDetectionA.ts 요약:")
    print(f"· 총 데이터 수           : {total}개")
    print(f"· 시계열 길이 종류       : {unique_lengths}")
    if len(unique_lengths) == 1:
        print(f"✅ 모든 데이터의 길이가 동일합니다. 길이 = {unique_lengths.pop()}")
    else:
        print("❗ 데이터의 길이가 서로 다릅니다.")

    print("\n· 클래스별 분포:")
    for label in sorted(label_counter):
        count = label_counter[label]
        ratio = count / total * 100
        print(f"  - 클래스 {label} : {count}개 ({ratio:.2f}%)")

# 사용
analyze_ts_multiclass("/home/hschoi/data/leehyunwon/time_series_FaultDetectionA/FaultDetectionA_TRAIN.ts")
