import numpy as np
import os 
import torch
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self, verbose=False, delta=0, use_gru=False):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.train_losses = []
        self.vali_losses = []
        self.use_gru = use_gru

    def __call__(self, train_loss, val_loss, model, path):
        self.train_losses.append(train_loss)
        self.vali_losses.append(val_loss)

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

    def save_checkpoint(self, val_loss, model, path):
        if not os.path.exists(path):
            os.makedirs(path)
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + ('checkpoint2.pth' if self.use_gru else 'checkpoint.pth'))
        self.val_loss_min = val_loss

    def plot_losses(self,save_path):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', color='blue')
        plt.plot(epochs, self.vali_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig(save_path)
