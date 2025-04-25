def count_cinc_labels(ts_file_path):
    normal_count = 0
    abnormal_count = 0

    with open(ts_file_path, 'r') as f:
        for line in f:
            # 메타정보나 주석은 건너뜀
            if line.startswith("@") or line.startswith("#") or line.strip() == "":
                continue

            parts = line.strip().split(":")
            if len(parts) < 2:
                continue  # 잘못된 줄 스킵

            label_str = parts[-1].strip().lower()
            if label_str == "normal":
                normal_count += 1
            elif label_str == "abnormal":
                abnormal_count += 1

    print(f"✅ 정상 (normal): {normal_count}개")
    print(f"❌ 비정상 (abnormal): {abnormal_count}개")
    print(f"📦 총 샘플 수: {normal_count + abnormal_count}개")


count_cinc_labels("/home/hschoi/data/leehyunwon/time_series_Heartbeat/Heartbeat_TRAIN.ts")