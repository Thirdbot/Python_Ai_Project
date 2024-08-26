#this transformer model just make it condition like datasets_loader for use in main file make init and its function
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datasets_loader import *
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformer_core import Transformer
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.set_default_device('cuda')
#simple model embeddings all ready got attention from tokenizer
#alright i gotta change it to gpt model but fast attention



class Transformers:
    def __init__(self) -> None:
        super().__init__()
        #its an attention of srcand tgt size that interpret so word_size need to be same as vocabb_size
        self.src_vocab_size = 25000
        self.tgt_vocab_size = 25000
        self.d_model = 768
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        # max_seq_length = 100
        self.dropout = 0.1
        self.lr = 0.0001
        self.word_size = 25000
        
        self.n_epochs = 100
        self.batch = 16 #batch in this refer to batch for training

        self.transformer = Transformer(self.src_vocab_size, self.tgt_vocab_size, self.d_model, self.num_heads, self.num_layers, self.d_ff, self.word_size, self.dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        


    def load_model(self, path):
        checkpoint = torch.load(path)
        
        # Load model state
        self.transformer.load_state_dict(checkpoint['model_state'])
        
        # Optionally load optimizer state if available
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Update internal attributes
        self.word_size = checkpoint.get('word_size', self.word_size)
        self.d_ff = checkpoint.get('hiddensize', self.d_ff)
        self.d_model = checkpoint.get('ndim', self.d_model)
        
        print(f'Model loaded from {path}')
        return self.transformer

    def save_model(self, path):
        # Save model state
        torch.save({
            'model_state': self.transformer.state_dict(),
            'optimizer_state': self.optimizer.state_dict() ,
            'word_size': self.word_size,
            'hiddensize': self.d_ff,
            'ndim': self.d_model,
        }, path)
        #print(f'Model saved to {path}')

    def test_input(self):
        file = "model_checkpoint.pth"
        self.transformer = self.load_model(file)
        self.transformer.eval()  # Set model to evaluation mode
        datasetss = Datasets()
        empty = ''
        while True:
            sentence = input("You: ")
            if sentence.lower() == "quit":
                break
            sentence = sentence.split()
            print(sentence)
            embbed_sent = self.ListEmbeddings(sentence,100)
            print(embbed_sent)
            embbed_sent = embbed_sent.to("cuda")  # Ensure embeddings are on the correct device
            print(embbed_sent.shape)
            # Initialize the tgt_data with start tokens, like    [CLS] or any start token you used during training
            tgt_data = torch.full((embbed_sent.shape[0], 1), 1, dtype=torch.long).to("cuda")
            for i in range(1, 100):  # Assuming max length 100
                
                with torch.no_grad():
                    output = self.transformer(embbed_sent, tgt_data)
                
                # Get the most likely next token
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

                # Append the predicted token to the target sequence
                tgt_data = torch.cat((tgt_data, next_token), dim=1)
                
                # Stop if the model predicts the end token
                if next_token.item() == 0:
                    break

            # Decode the generated sequence
            output_tokens = tgt_data.squeeze().tolist()
            decoded_output = datasetss.decode(output_tokens)
            print("output: ", decoded_output)

            fig = plt.figure()
            images = self.transformer.decoder_layers[0].cross_attn[0,...].cpu().detach().numpy().mean(axis=0)

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            ax.set_yticks(range(len(output)))
            ax.set_xticks(range(len(sentence)))
            ax.xaxis.set_label_position('top')
            ax.set_xticklabels(list(sentence))
            ax.set_yticklabels([f"step {i}" for i in range(len(output))])
            images = np.clip(images, 0, 1)
            # images = np.mean(images, axis=0)
            cax = ax.imshow(images, aspect='auto', cmap='viridis')
            fig.colorbar(cax)
            plt.show()  # Ensure the plot is displayed

            
    def runtrain(self,inputs,outs):
        loss = self.feedmodel(inputs,outs)
        return loss

    def ListEmbeddings(self,list_input,word_size):
        datasets_Detail = Datasets()
        embeds = []
        sequence_lengths = []

        embed_input = datasets_Detail.set_tokenizer.batch_encode_plus(list_input, padding='max_length',truncation=True,add_special_tokens = False,return_attention_mask = True,max_length=word_size, return_tensors='pt')
        
        embedded_model = GPT2Model.from_pretrained('gpt2')
        tensor_id = embed_input['input_ids'].long()
        tensor_mask = embed_input['attention_mask'].long()
        
        return tensor_id
       


    
    
    def feedmodel(self,list_input,list_output):
        
       
        self.transformer.train()
        losses = 0
        acc = 0
        history_loss = []
        history_acc = []
        history = {
            'train_loss': [],
            'eval_loss': [],
            'train_acc': [],
            'eval_acc': []
        }

        # src_data = torch.randint(1, 25000, (1, 100))  # (batch_size, seq_length)
        # tgt_data = torch.randint(1, 25000, (1, 100))  # (batch_size, seq_length)
        # datasetss = Datasets()
        with tqdm(range(self.n_epochs), position=0, leave=False) as tepoch:
            for epochs in tepoch:
                start_time = time.time()
                self.optimizer.zero_grad()
                tepoch.set_description(f"Epoch {epochs}")
                 #for epochs in tqdm(range(self.n_epochs),desc="EPOCHS:",leave=False):
                count = 0
                # size = 0
                with tqdm(zip(list_input,list_output), position=0, leave=False) as tbatch:
                    for (list_in,list_out) in tbatch:

                        # print(f"list_in size: {len(list_in)} list_out size: {len(list_out)}")
                        #batch data again
                        tepoch.set_description(f"BATCHES {count}")

                        input_loader = DataLoader(list_in, batch_size=self.batch, num_workers=0)
                        output_loader = DataLoader(list_out, batch_size=self.batch, num_workers=0)
                        
                        # self.model.optimizer.zero_grad()

                        for list_inin, list_outout in zip(input_loader,output_loader):
                            
                            
                            list_inin = list_inin.to("cuda")
                            list_outout = list_outout.to("cuda")

                            # print(f"\tlist_inin size: {list_inin.shape} list_outout size: {list_outout.shape}")
                            #print(list_inin,list_outout[:,:1])
                            # print(list_inin.shape,list_outout.shape)
                            # print(src_data,tgt_data)
                            # qdecode = datasetss.decode(list_inin)
                            # adecode = datasetss.decode(list_outout)
                            # print(f"Question: {qdecode}\nAnswer{adecode}")
                            output = self.transformer(list_inin, list_outout[:,:-1])
                            #output = self.transformer(list_inin, list_outout)

                        
                            
                            # loss = criterion(output.contiguous().view(-1, self.tgt_vocab_size), list_outout[:, 1:].contiguous().view(-1))
                            loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size), list_out[:, 1:].contiguous().view(-1))
                            loss.backward()

                            losses += loss.item()

                            preds = output.argmax(dim=-1)
                            masked_pred = preds * (list_outout[:, 1:]!=0)
                            accuracy = (masked_pred == list_outout[:, 1:]).float().mean()
                            acc += accuracy.item()

                            self.optimizer.step()

                            history_loss.append(loss.item())
                            history_acc.append(accuracy.item())
                            train_loss, train_acc, hist_loss, hist_acc = losses / len(list(list_out)), acc / len(list(list_out)), history_loss, history_acc
                            history['train_loss'] += hist_loss
                            history['train_acc'] += hist_acc

    
                            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy.item())
                            #print(f"Epoch: {epochs+1}, Loss: {loss.item()}")
                            count += 1
        
                            end_time = time.time()
                        val_loss, val_acc, hist_loss, hist_acc = self.evaluate(self.transformer, zip(list_in, list_out), self.criterion)
                        history['eval_loss'] += hist_loss
                        history['eval_acc'] += hist_acc
                        print((f"Epoch: {epochs}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))

                model_save_path = "model_checkpoint.pth"
                self.save_model(model_save_path)
            
    def evaluate(self,model, loader, loss_fn):
        model.eval()
        losses = 0
        acc = 0
        history_loss = []
        history_acc = [] 

        for x, y in tqdm(loader, position=0, leave=True):
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            logits = model(x, y[:,:-1])
            loss = loss_fn(logits.contiguous().view(-1, self.tgt_vocab_size), y[:, 1:].contiguous().view(-1))
            losses += loss.item()
            
            preds = logits.argmax(dim=-1)
            masked_pred = preds * (y[:, 1:]!=0)
            accuracy = (masked_pred == y[:, 1:]).float().mean()
            acc += accuracy.item()
            
            history_loss.append(loss.item())
            history_acc.append(accuracy.item())

        return losses / len(list(y)), acc / len(list(y)), history_loss, history_acc

                    
                #     size += len(list_inin)
                # print(f"batch done total size {size}")
            
                    
# Credit to Arjun Sarkar and my tweak            
                    
                    
                    

        

###validation goes here back prob goes heere

if __name__ == "__main__":
    transformers = Transformers()