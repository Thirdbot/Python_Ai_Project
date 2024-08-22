from language_model import *
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from mpl_toolkits.axes_grid1 import ImageGrid
from language_model import Transformer

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def train(model, optimizer, loader, loss_fn, epoch):
    model.train()
    losses = 0
    acc = 0
    history_loss = []
    history_acc = [] 

    with tqdm(loader, position=0, leave=True) as tepoch:
        for x, y in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            optimizer.zero_grad()
            logits = model(x, y[:, :-1])
            loss = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
            preds = logits.argmax(dim=-1)
            masked_pred = preds * (y[:, 1:]!=PAD_IDX)
            accuracy = (masked_pred == y[:, 1:]).float().mean()
            acc += accuracy.item()
            
            history_loss.append(loss.item())
            history_acc.append(accuracy.item())
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy.item())

    return losses / len(list(loader)), acc / len(list(loader)), history_loss, history_acc


def evaluate(model, loader, loss_fn):
    model.eval()
    losses = 0
    acc = 0
    history_loss = []
    history_acc = [] 

    for x, y in tqdm(loader, position=0, leave=True):

        logits = model(x, y[:, :-1])
        loss = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
        losses += loss.item()
        
        preds = logits.argmax(dim=-1)
        masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        accuracy = (masked_pred == y[:, 1:]).float().mean()
        acc += accuracy.item()
        
        history_loss.append(loss.item())
        history_acc.append(accuracy.item())

    return losses / len(list(loader)), acc / len(list(loader)), history_loss, history_acc

np.random.seed(0)

def generate_random_string():
    len = np.random.randint(10, 20)
    return "".join([chr(x) for x in np.random.randint(97, 97+26, len)])


class ReverseDataset(Dataset):
    def __init__(self, n_samples, pad_idx, sos_idx, eos_idx):
        super(ReverseDataset, self).__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.values = [generate_random_string() for _ in range(n_samples)]
        self.labels = [x[::-1] for x in self.values]

    def __len__(self):
        return len(self.values)  # number of samples in the dataset

    def __getitem__(self, index):
        return self.text_transform(self.values[index].rstrip("\n")), \
            self.text_transform(self.labels[index].rstrip("\n"))
        
    def text_transform(self, x):
        return torch.tensor([self.sos_idx] + [ord(z)-97+3 for z in x] + [self.eos_idx])
    
def collate_fn(batch):
    """ 
    This function pads inputs with PAD_IDX to have batches of equal length
    """
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

# Model hyperparameters

class TrainInterference:
    def __init__(self) -> None:
        self.args = {
            'vocab_size': 100,
            'model_dim': 768,
            'dropout': 0.1,
            'n_encoder_layers': 1,
            'n_decoder_layers': 1,
            'n_heads': 4
        }

        # Define model here
        self.model = Transformer(**self.args)

        # Instantiate datasets
        self.train_iter = ReverseDataset(50000, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
        self.eval_iter = ReverseDataset(10000, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
        self.dataloader_train = DataLoader(self.train_iter,batch_size=4, collate_fn=collate_fn)
        self.dataloader_val = DataLoader(self.eval_iter,batch_size=4, collate_fn=collate_fn)

        # During debugging, we ensure sources and targets are indeed reversed
        # s, t = next(iter(dataloader_train))
        # print(s[:4, ...])
        # print(t[:4, ...])
        # print(s.size())

        # Initialize model parameters
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Define loss function : we ignore logits which are padding tokens
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

        # Save history to dictionnary
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'train_acc': [],
            'eval_acc': []
        }

    def runtrain(self,dt,dv):
        for epoch in tqdm(range(1,10),desc='Epoch',leave=False):
            start_time = time.time()
            train_loss, train_acc, hist_loss, hist_acc = train(self.model, self.optimizer, dt, self.loss_fn, epoch)
            self.history['train_loss'] += hist_loss
            self.history['train_acc'] += hist_acc
            end_time = time.time()
            val_loss, val_acc, hist_loss, hist_acc = evaluate(self.model, dv, self.loss_fn)
            self.history['eval_loss'] += hist_loss
            self.history['eval_acc'] += hist_acc
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))
        # fig = plt.figure(figsize=(10., 10.))
        # images = self.model.decoder.decoder_blocks[0].cross_attention.attention_weigths[0,...].detach().numpy()
        # grid = ImageGrid(fig, 111,  # similar to subplot(111)
        #                 nrows_ncols=(2, 2),  # creates 2x2 grid of axes
        #                 axes_pad=0.1,  # pad between axes in inch.
        #                 )

        # for ax, im in zip(grid, images):
        #     # Iterating over the grid returns the Axes.
        #     ax.imshow(im)
        
        return self.history['train_loss']
    
    def runpredict(self,x,max_length):
        output = self.model.predict(x,max_length)
        return output
            
if __name__ == "__main__":
    training = TrainInterference()