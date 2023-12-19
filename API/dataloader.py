import os
from typing import List
import gdown
import cv2
import torch
import numpy as np
import dlib

def get_data():
    if not os.path.exists('data'):
        url = 'https://drive.google.com/u/1/uc?id=173NlesZaWfaG1atByzDCKL0Zi0lNzcxG'
        output = 'data.zip'
        gdown.download(url, output, quiet=False)
        gdown.extractall('data.zip')

def load_video(path: str, apply_mouth_detection: bool = False) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not apply_mouth_detection:
            frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frame = frame[190:236, 80:220]
            frames.append(frame.numpy())
        else:
            frame = mouth_detector(frame)
            frames.append(frame)
        

    cap.release()

    frames = np.array(frames)
    mean = np.mean(frames)
    std = np.std(frames, dtype=np.float32)
    frames = (frames - mean) / std

    return torch.from_numpy(frames)

def mouth_detector(frame_test):
  shape_predictor_path = "./pretrain/dlib/shape_predictor_68_face_landmarks.dat"
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(shape_predictor_path)
  MOUTH_WIDTH = 140
  MOUTH_HEIGHT = 46
  HORIZONTAL_PAD = 0.19

  frame = cv2.cvtColor(frame_test, cv2.COLOR_BGR2GRAY)
  dets = detector(frame)

  shape = None
  for k, d in enumerate(dets):
      shape = predictor(frame, d)
      i = -1

  if shape is None:
      return frames
  mouth_points = []

  for part in shape.parts():
      i += 1
      if i < 48:
          continue
      mouth_points.append((part.x, part.y))

  np_mouth_points = np.array(mouth_points)
  mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

  mouth_l = int(mouth_centroid[0] - MOUTH_WIDTH / 2)
  mouth_r = int(mouth_centroid[0] + MOUTH_WIDTH / 2)
  mouth_t = int(mouth_centroid[1] - MOUTH_HEIGHT / 2)
  mouth_b = int(mouth_centroid[1] + MOUTH_HEIGHT / 2)
  mouth_crop_image = frame[mouth_t:mouth_b, mouth_l:mouth_r]

  return mouth_crop_image
