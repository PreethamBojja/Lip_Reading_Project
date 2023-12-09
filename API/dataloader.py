import os
from typing import List
import gdown
import cv2
import torch
import numpy as np

def get_data():
    if not os.path.exists('data'):
        url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
        output = 'data.zip'
        gdown.download(url, output, quiet=False)
        gdown.extractall('data.zip')

def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frame = frame[190:236, 80:220]
        frames.append(frame.numpy())

    cap.release()

    frames = np.array(frames)
    mean = np.mean(frames)
    std = np.std(frames, dtype=np.float32)
    frames = (frames - mean) / std

    return torch.from_numpy(frames)