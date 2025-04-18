import torch
from torch.utils.data import (DataLoader, Dataset)  # 미니배치 등의 데이터셋 관리를 도와주는 녀석
from typing import Callable # 람다식
import pandas as pd # MIT-BIH .csv 읽기용
import numpy as np # CinC .ts 읽기용

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
    


# CinC Loader
class CinC_Loader(Dataset):
    def __init__(self, ts_file_path, transforms=lambda x: x):
        super().__init__()
        self.data = []
        self.labels = []
        self.transforms = transforms

        with open(ts_file_path, 'r') as f:
            for line in f:
                if line.startswith("@") or line.strip() == "":
                    continue
                values, label = line.strip().split(":")
                features = np.array([float(v) for v in values.split(",")], dtype=np.float32)
                label = 0 if label.strip().lower() == "normal" else 1

                # Min-max normalization to [0, 1]
                features = (features - features.min()) / (features.max() - features.min() + 1e-8)

                self.data.append(features)
                self.labels.append(label)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transforms:
            x = self.transforms(x)
        y = self.labels[idx]
        return x, y