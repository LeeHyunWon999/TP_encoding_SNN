import json

def count_data_points(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 각 라벨의 개수를 셈
    label_counts = {0: 0, 1: 0}
    for label in data.values():
        if label in label_counts:
            label_counts[label] += 1
    
    total_count = len(data)
    return total_count, label_counts[0], label_counts[1]

# 사용 예시
json_file_path = '/home/hschoi/data/leehyunwon/CinC_original_Heartbeat_audio_preprocessed_train/label_dict.json'
total, count_0, count_1 = count_data_points(json_file_path)
print(f"Total data points: {total}")
print(f"Label 0 count: {count_0}")
print(f"Label 1 count: {count_1}")
