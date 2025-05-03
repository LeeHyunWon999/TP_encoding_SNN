from collections import Counter

def analyze_fruitflies_ts(ts_file_path):
    with open(ts_file_path, 'r') as f:
        lines = f.readlines()

    # @data 이후의 유효한 데이터만 추출
    data_start_idx = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
    data_lines = [line.strip() for line in lines[data_start_idx:] if line.strip()]

    label_counter = Counter()
    data_lengths = []

    for line in data_lines:
        if ':' not in line:
            continue
        signal_str, label_str = line.rsplit(':', 1)
        signal = signal_str.split(',')
        label = int(label_str.strip())

        data_lengths.append(len(signal))
        label_counter[label] += 1

    total = len(data_lines)
    unique_lengths = set(data_lengths)

    print("📊 FruitFlies 데이터셋 요약:")
    print(f"· 총 데이터 수           : {total}개")
    print(f"· 시계열 길이 종류       : {unique_lengths}")
    if len(unique_lengths) == 1:
        print(f"✅ 모든 시계열의 길이가 동일합니다. 길이 = {unique_lengths.pop()}")
    else:
        print("❗ 서로 다른 길이의 시계열이 존재합니다.")

    print("\n· 클래스별 분포:")
    for label in sorted(label_counter):
        count = label_counter[label]
        ratio = count / total * 100
        print(f"  - 클래스 {label} : {count}개 ({ratio:.2f}%)")

# 사용
analyze_fruitflies_ts("/home/hschoi/data/leehyunwon/time_series_FruitFlies/FruitFlies_TRAIN.ts")
