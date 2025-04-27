import json

def count_data_points(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return len(data)

# 사용 예시
json_file_path = '/home/hschoi/data/leehyunwon/CinC_original_Heartbeat_audio_preprocessed_train/label_dict.json'
print(count_data_points(json_file_path))
