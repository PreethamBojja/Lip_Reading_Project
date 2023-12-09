import gdown
import os
import torch
import torch.nn as nn
from torch.nn.functional import pad
import cv2
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio
from torchinfo import summary
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from glob import glob

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

# We'll be predicting character by character
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

def char_to_num(chars):
  if isinstance(chars, str):
      return vocab.index(chars) + 1
  elif isinstance(chars, list):
      return [vocab.index(char) + 1 for char in chars]
  else:
      raise ValueError("Input must be a single character or a list of characters")

def num_to_char(indices):
  if isinstance(indices, int):
      return vocab[indices - 1]
  elif isinstance(indices, list):
      return [vocab[index - 1] for index in indices]
  else:
      raise ValueError("Input must be a single index or a list of indices.")

def padding(array, length):
      array = [array[_] for _ in range(array.shape[0])]
      size = array[0].shape
      for i in range(length - len(array)):
          array.append(np.zeros(size))
      return torch.from_numpy(np.stack(array, axis=0))

def load_alignments(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()

    tokens = []

    for line in lines:
        line = line.split()
        if line[2] != 'sil': #ignoring the initial silence in the video
            tokens += [' ', line[2]]

    # dummy = [char for word in tokens for char in word]
    # print(torch.tensor(char_to_num(dummy), dtype=torch.long))
    tokens_tensor = torch.tensor(char_to_num([char for word in tokens for char in word]), dtype=torch.long)
    alignment = tokens_tensor[1:] #ignoring the first space which is appended in the for loop
    return pad(alignment, (0, 40 - len(alignment))) #making all allignments of same size (40)

def load_data(path: str):
    path = path.item() if torch.is_tensor(path) else path
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    # file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path)
    frames = padding(frames,75)
    alignments = load_alignments(alignment_path)

    return frames.unsqueeze(-1), alignments

def custom_collate(batch):
    data, labels = zip(*batch)
    data_dum = [torch.zeros((2, 2)) for _ in data]
    for new_tensor, original_tensor in zip(data_dum, data):
        new_tensor[:original_tensor.numel()] = original_tensor
    return data_dum, labels

class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data, label = load_data(path)
        return data, label

class CustomModel(nn.Module):
    def __init__(self, vocab_size):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv3d(1, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3 = nn.Conv3d(256, 75, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.flatten = nn.Flatten()
        self.time_dist = nn.Sequential(nn.Linear(75 * 5 * 17, 128), nn.ReLU())

        self.lstm1 = nn.LSTM(128, 128, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(256, 128, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)

        self.dense = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # x = self.flatten(x)
        size = x.size()
        x = x.view(size[0], size[1], -1)
        x = self.time_dist(x)

        x, _ = self.lstm1(x.permute(1,0,2))

        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x = self.dense(x)

        return x

class CTCLoss(nn.Module):
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean')

class ProduceExampleCallback:
    def __init__(self, dataset, num_to_char_fn) -> None:
        self.dataset = dataset
        self.num_to_char_fn = num_to_char_fn

    def on_epoch_end(self, epoch, model, device='cuda'):
        data = next(iter(self.dataset))
        inputs, targets = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            inputs = inputs.permute(3, 0, 1, 2).unsqueeze(0).float()
            outputs = model(inputs)
            log_probs = F.log_softmax(outputs, dim=2)
            argmax_result = torch.argmax(log_probs, dim=-1, keepdim=True)
            original = ''.join(self.num_to_char_fn(value) for value in targets.cpu().numpy().tolist())
            prediction = ''.join(self.num_to_char_fn(value) for value in argmax_result.cpu().squeeze(-1).squeeze(-1).numpy().tolist())
            print('Original:', original)
            print('Prediction:', prediction)
            print('~' * 100)

def execute():
    print("Got into execute")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_path = './data/s1/bbal6n.mpg'
    frames, alignments = load_data(test_path)

    plt.imshow(frames[29])

    alignments

    ''.join(num_to_char(alignments.numpy().tolist()))

    all_file_paths = glob('./data/s1/*.mpg')
    # Taking half dataset for now
    file_paths = all_file_paths[:500]
    dataset = CustomDataset(file_paths)

    train_size = int(0.9 * len(dataset))
    valid_size = int(0.09 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1)
    train_loader_iter = iter(train_loader)
    try:
        frames, allignments = next(train_loader_iter)
    except StopIteration:
        print("End of DataLoader reached")

    val = next(train_loader_iter)
    val[0][0].shape

    plt.imshow(val[0][0][35])

    ''.join(num_to_char(val[1][0].numpy().tolist()))

    val[0][0].shape

    vocab_size = len(vocab)+1
    model = CustomModel(vocab_size)

    print(model)
    summary(model, input_size=(1, 1, 75, 46, 140))

    tensor_permuted = val[0][0].permute(3, 0, 1, 2)
    tensor_permuted.shape

    model = model.float()
    for module in model.children():
        if isinstance(module, nn.LSTM):
            for param_name, param in module.named_parameters():
                if param.dtype == torch.bool:
                    module.param.data = module.param.data.float()
        else:
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.dtype == torch.bool:
                    module.weight.data = module.weight.data.float()

            if hasattr(module, 'bias') and module.bias is not None:
                if module.bias.dtype == torch.bool:
                    module.bias.data = module.bias.data.float()

    model.eval()
    with torch.no_grad():
        output = model(tensor_permuted.unsqueeze(0).float().to(device))
    # print("Predictions:", output)
    print("Prediction Shape:", output.shape)


    # Assume Adam optimizer and dataset loaders `train_loader` and `test_loader` are defined
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 if epoch < 30 else torch.exp(torch.tensor(-0.1)))

    ctc_loss = CTCLoss()

    # Assume num_to_char function is defined
    example_callback = ProduceExampleCallback(test_dataset, num_to_char)

    # Training loop
    num_epochs = 1
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            inputs = inputs.permute(0, 4, 1, 2, 3).float()

            outputs = model(inputs)

            log_probs = F.log_softmax(outputs, dim=2)

            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
            target_lengths = torch.full(size=(targets.size(0),), fill_value=targets.size(1), dtype=torch.long)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        scheduler.step()

        # Validation
        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            for data in tqdm(valid_loader, desc="Validation"):
                inputs, targets = data[0].to(device), data[1].to(device)
                inputs = inputs.permute(0, 4, 1, 2, 3).float()
                outputs = model(inputs)
                log_probs = F.log_softmax(outputs, dim=2)

                input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
                target_lengths = torch.full(size=(targets.size(0),), fill_value=targets.size(1), dtype=torch.long)

                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_valid_loss:.4f}")

        example_callback.on_epoch_end(epoch, model, device)



if __name__ == "__main__":
    print("got into main")
    get_data()
    execute()