import os
import torch
from torch.utils.data import (DataLoader, Dataset)  # 미니배치 등의 데이터셋 관리를 도와주는 녀석
from typing import Callable # 람다식
import pandas as pd # MIT-BIH .csv 읽기용
import numpy as np # CinC .ts 읽기용
import json # CinC_original .json 읽기용

# MIT-BIH Loader
class MITLoader_MLP_binary(Dataset):

    def __init__(self, csv_file, transforms: Callable = lambda x: x) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file).values
        self.transforms = transforms

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, item):
        signal = self.annotations[item, :-1]
        signal = torch.from_numpy(signal).float()
        if self.transforms : 
            signal = self.transforms(signal)
        
        label = int(self.annotations[item, -1])
        if label > 0:
            label = 1  # 1 이상인 모든 값은 1로 변환(난 이진값 처리하니깐)

        return signal, torch.tensor(label).long()
    


# CinC_simpled Loader
class CinC_Loader(Dataset):
    def __init__(self, ts_file_path, normalize=True, transforms=lambda x: x):
        super().__init__()
        self.data = []
        self.labels = []
        self.transforms = transforms
        self.normalize = normalize

        with open(ts_file_path, 'r') as f:
            for line in f:
                if line.startswith("@") or line.startswith("#") or line.strip() == "":
                    continue

                parts = line.strip().split(":")
                *channel_strs, label_str = parts

                # 각 채널을 파싱 (405개 타임스텝 가정)
                channel_data = []
                for ch_str in channel_strs:
                    ch_vals = np.array([float(v) for v in ch_str.strip().split(",")], dtype=np.float32)
                    if self.normalize:
                        ch_vals = (ch_vals - ch_vals.min()) / (ch_vals.max() - ch_vals.min() + 1e-8)
                    channel_data.append(ch_vals)

                sample = np.stack(channel_data, axis=0)  # shape: (61, T)
                label = 0 if label_str.strip().lower() == "normal" else 1

                self.data.append(sample)
                self.labels.append(label)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # shape: (61, T)
        if self.transforms:
            x = self.transforms(x)
        y = self.labels[idx]
        return x.view(-1), y  # ← 기본은 flatten해서 반환


# CinC_original Loader
class CinC_original_Loader(Dataset):
    def __init__(self, npy_dir, label_json_path):
        self.npy_dir = npy_dir

        # load label dictionary
        with open(label_json_path, "r") as f:
            self.label_dict = json.load(f)

        self.file_ids = list(self.label_dict.keys())

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        npy_path = os.path.join(self.npy_dir, file_id + ".npy")

        data = np.load(npy_path)  # shape: [6, 500]
        data_flat = data.flatten()  # shape: [3000]
        label = self.label_dict[file_id]

        return torch.tensor(data_flat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# gesture Loader
class Gesture_Loader(Dataset):
    def __init__(self, ts_file_path):
        self.data = []
        self.labels = []

        with open(ts_file_path, 'r') as f:
            lines = f.readlines()

        # 데이터 라인 추출
        data_start_idx = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
        data_lines = [line.strip() for line in lines[data_start_idx:] if line.strip()]

        for line in data_lines:
            if ':' in line:
                signal_str, label_str = line.rsplit(':', 1)
                signal = np.array([float(x) for x in signal_str.split(',')], dtype=np.float32)

                # 0~1 정규화 (각 데이터포인트 기준)
                min_val, max_val = np.min(signal), np.max(signal)
                if max_val - min_val > 0:
                    signal = (signal - min_val) / (max_val - min_val)
                else:
                    signal = np.zeros_like(signal)  # 상수 시계열 처리

                self.data.append(signal)
                self.labels.append(int(label_str.strip()) - 1)  # 1~8 → 0~7

        self.data = [torch.tensor(d) for d in self.data]
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

# fordA Loader
class FordA_Loader(Dataset):
    def __init__(self, ts_file_path):
        self.data = []
        self.labels = []

        with open(ts_file_path, 'r') as f:
            lines = f.readlines()

        # @data 이후부터 유효 데이터 라인
        data_start_idx = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
        data_lines = [line.strip() for line in lines[data_start_idx:] if line.strip()]

        for line in data_lines:
            if ':' in line:
                signal_str, label_str = line.rsplit(':', 1)
                signal = np.array([float(x) for x in signal_str.split(',')], dtype=np.float32)

                # 0~1로 정규화 (데이터포인트 단위)
                min_val, max_val = np.min(signal), np.max(signal)
                if max_val - min_val > 0:
                    signal = (signal - min_val) / (max_val - min_val)
                else:
                    signal = np.zeros_like(signal)

                # 라벨 정규화: -1 → 0, 1 → 1
                label = 0 if int(label_str.strip()) == -1 else 1

                self.data.append(torch.tensor(signal))
                self.labels.append(label)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# faultD Loader
class FaultD_Loader(Dataset):
    def __init__(self, ts_file_path):
        self.data = []
        self.labels = []

        with open(ts_file_path, 'r') as f:
            lines = f.readlines()

        # @data 이후부터 유효한 데이터
        data_start_idx = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
        data_lines = [line.strip() for line in lines[data_start_idx:] if line.strip()]

        for line in data_lines:
            if ':' not in line:
                continue

            signal_str, label_str = line.rsplit(':', 1)
            signal = np.array([float(x) for x in signal_str.split(',')], dtype=np.float32)

            # 0~1 정규화 (데이터포인트 단위)
            min_val, max_val = np.min(signal), np.max(signal)
            if max_val - min_val > 0:
                signal = (signal - min_val) / (max_val - min_val)
            else:
                signal = np.zeros_like(signal)

            label = int(label_str.strip())

            self.data.append(torch.tensor(signal))  # shape: [5120]
            self.labels.append(label)               # int: 0, 1, 2

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# fruitfly Loader
class FruitFly_Loader(Dataset):
    def __init__(self, ts_file_path):
        self.data = []
        self.labels = []

        with open(ts_file_path, 'r') as f:
            lines = f.readlines()

        # @data 이후 유효 데이터 시작
        data_start_idx = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
        data_lines = [line.strip() for line in lines[data_start_idx:] if line.strip()]

        for line in data_lines:
            if ':' not in line:
                continue

            signal_str, label_str = line.rsplit(':', 1)
            signal = np.array([float(x) for x in signal_str.split(',')], dtype=np.float32)

            # percentile 기반 정규화
            lower = np.percentile(signal, 1)
            upper = np.percentile(signal, 99)

            signal = np.clip(signal, lower, upper)
            if upper - lower > 0:
                signal = (signal - lower) / (upper - lower)
            else:
                signal = np.zeros_like(signal)

            label = int(label_str.strip())

            self.data.append(torch.tensor(signal))  # shape: [5000]
            self.labels.append(label)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]