# CinC 오리지널 심박음 데이터셋 전처리
# - .wav 6개 폴더 통합
# - 5초로 길이 통일(주로 자르기; 제로패딩), fps 100 (즉 500프레임), 주파수 6개 채널로 6*500의 멜스펙트로그램화, 0~1로 normalize
# - .npy로 라벨 정보와 함께 변환하여 저장

import os
import csv
import json
import torchaudio
import torchaudio.transforms as T
import torch
import numpy as np
from tqdm import tqdm

class HeartSoundPreprocessor:
    def __init__(self, target_sample_rate=2000, target_duration=5, n_mels=6, target_frames=500):
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.target_num_samples = int(target_sample_rate * target_duration)
        self.n_mels = n_mels
        self.target_frames = target_frames

        self.hop_length = self._calc_hop_length()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=2048,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

    def _calc_hop_length(self):
        total_hops = self.target_frames - 1
        return self.target_num_samples // total_hops

    def _normalize(self, spec):
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max > spec_min:
            return (spec - spec_min) / (spec_max - spec_min)
        else:
            return spec - spec_min

    def process_wav_to_npy(self, wav_path, save_path):
        try:
            waveform, sr = torchaudio.load(wav_path)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != self.target_sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[1] < self.target_num_samples:
                padding = self.target_num_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.shape[1] > self.target_num_samples:
                start = (waveform.shape[1] - self.target_num_samples) // 2
                waveform = waveform[:, start:start+self.target_num_samples]

            mel_spec = self.mel_transform(waveform).squeeze(0)

            if mel_spec.shape[1] < self.target_frames:
                pad_amt = self.target_frames - mel_spec.shape[1]
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amt))
            elif mel_spec.shape[1] > self.target_frames:
                mel_spec = mel_spec[:, :self.target_frames]

            mel_spec = self._normalize(mel_spec)

            np.save(save_path, mel_spec.numpy())
        except Exception as e:
            print(f"❌ Error processing {wav_path}: {e}")

def load_labels_from_reference(csv_path):
    label_dict = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            file_id, label = row
            label = 0 if int(label) == -1 else 1
            label_dict[file_id] = label
    return label_dict

def batch_process_all_wavs_with_labels(raw_root, output_dir, subfolders=['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']):
    preprocessor = HeartSoundPreprocessor()
    os.makedirs(output_dir, exist_ok=True)

    full_label_dict = {}

    for subdir in subfolders:
        folder_path = os.path.join(raw_root, subdir)
        ref_path = os.path.join(folder_path, "REFERENCE.csv")

        if not os.path.exists(ref_path):
            print(f"⚠️ Missing REFERENCE.csv in {folder_path}")
            continue

        label_dict = load_labels_from_reference(ref_path)

        print(f"📂 Processing {subdir} ({len(label_dict)} files)")
        for file_id, label in tqdm(label_dict.items()):
            wav_path = os.path.join(folder_path, file_id + ".wav")
            if not os.path.exists(wav_path):
                print(f"⚠️ Missing .wav file: {wav_path}")
                continue

            save_path = os.path.join(output_dir, file_id + ".npy")
            preprocessor.process_wav_to_npy(wav_path, save_path)
            full_label_dict[file_id] = label

    # Save label dictionary
    label_json_path = os.path.join(output_dir, "label_dict.json")
    with open(label_json_path, "w") as f:
        json.dump(full_label_dict, f)

    print(f"✅ All .wav files processed and saved to {output_dir}")
    print(f"✅ Labels saved to {label_json_path}")





# 실행 부분
raw_root = "/home/hschoi/data/leehyunwon/CinC_original_Heartbeat_audio"  # training-a ~ training-f가 들어있는 상위 폴더
output_dir = "/home/hschoi/data/leehyunwon/CinC_original_Heartbeat_audio_preprocessed"

batch_process_all_wavs_with_labels(raw_root, output_dir)
