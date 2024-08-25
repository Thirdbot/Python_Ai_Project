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
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        # max_seq_length = 100
        self.dropout = 0.1

        self.word_size = 25000
        # self.hiddensize = 256
        # self.ndim = 768
        # self.lr = 0.000001
        #self.num_layers = 2 #bidirectional
        self.n_epochs = 100
        self.batch = 4 #batch in this refer to batch for training

        self.transformer = Transformer(self.src_vocab_size, self.tgt_vocab_size, self.d_model, self.num_heads, self.num_layers, self.d_ff, self.word_size, self.dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        


    def load_model(self, path):
        checkpoint = torch.load(path)
        
        # Load model state
        self.transformer.load_state_dict(checkpoint['model_state'])
        
        # Optionally load optimizer state if available
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Update internal attributes
        self.word_size = checkpoint.get('word_size', self.word_size)
        self.d_ff = checkpoint.get('hiddensize', self.hiddensize)
        self.d_model = checkpoint.get('ndim', self.ndim)
        
        print(f'Model loaded from {path}')
        return self.transformer

    def save_model(self, path):
        # Save model state
        torch.save({
            'model_state': self.transformer.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if hasattr(self.transformer, 'optimizer') else None,
            'word_size': self.word_size,
            'hiddensize': self.d_ff,
            'ndim': self.d_model,
        }, path)
        #print(f'Model saved to {path}')

    # def test_input(self):
    #     file = "model_checkpoint.pth"
    #     self.transformer = self.load_model(file)
    #     self.transformer.eval()  # Set model to evaluation mode
    #     datasetss = Datasets()
    #     empty = ''
    #     while True:
    #         sentence = input("You: ")
    #         if sentence.lower() == "quit":
    #             break

    #         embbed_sent = self.ListEmbeddings(sentence, self.word_size)
    #         embbed_sent = embbed_sent.to("cuda")  # Ensure embeddings are on the correct device

    #         with torch.no_grad():
    #             # Assuming Translator is a method/function of the model
    #             output = test.Translator(self.transformer)(embbed_sent, max_length=self.word_size)
        
    #         empty.join(datasetss.decode([output]))
    #         print("output: ",empty)

    #         fig = plt.figure()
    #         images = self.transformer.decoder.decoder_blocks[0].cross_attention.attention_weigths[0,...].cpu().detach().numpy().mean(axis=0)

    #         fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
    #         ax.set_yticks(range(len(output)))
    #         ax.set_xticks(range(len(sentence)))
    #         ax.xaxis.set_label_position('top')
    #         ax.set_xticklabels(list(sentence))
    #         ax.set_yticklabels([f"step {i}" for i in range(len(output))])
    #         images = np.clip(images, 0, 1)
    #         # images = np.mean(images, axis=0)
    #         cax = ax.imshow(images, aspect='auto', cmap='viridis')
    #         fig.colorbar(cax)
    #         plt.show()  # Ensure the plot is displayed

            
    def runtrain(self,inputs,outs):
        loss = self.feedmodel(inputs,outs)
        return loss

    def ListEmbeddings(self,list_input,word_size):
        datasets_Detail = Datasets()
        embeds = []
        sequence_lengths = []

        for input in list_input:
            embed_input = datasets_Detail.set_tokenizer.encode_plus(input, padding='max_length',truncation=True,add_special_tokens = True,return_attention_mask = True,max_length=word_size, return_tensors='pt')
            embedded_model = GPT2Model.from_pretrained('gpt2')
            tensor_id = embed_input['input_ids']
            tensor_mask = embed_input['attention_mask']
            # with torch.no_grad():
            #     q_output = embedded_model(tensor_id,attention_mask=tensor_mask)
            
            # embeds.append(q_output.last_hidden_state.squeeze(0))
            # embeds.append(tensor_id.tolist())
        # embeds = torch.stack(embeds)
        # sequence_lengths.append(q_output.last_hidden_state.size(1))
        sequence_lengths.append(tensor_id.shape[0])
        padded_embeds = rnn_utils.pad_sequence(tensor_id, batch_first=False, padding_value=0)
        if len(padded_embeds) < self.word_size:
            num_padding = self.word_size - len(padded_embeds)
            padding_tensors = torch.zeros((num_padding, padded_embeds.size(1), padded_embeds.size(2)))
            padded_embeds = torch.cat([padded_embeds, padding_tensors], dim=0)
            sequence_lengths.extend([0] * num_padding)
        else:
            padded_embeds = padded_embeds.transpose(0,1).to("cuda")
        return padded_embeds



    
    def feedmodel(self,list_input,list_output):
        
       
        self.transformer.train()
        losses = 0
        acc = 0
        history_loss = []
        history_acc = [] 
        # src_data = torch.randint(1, 25000, (1, 100))  # (batch_size, seq_length)
        # tgt_data = torch.randint(1, 25000, (1, 100))  # (batch_size, seq_length)
        # datasetss = Datasets()
        with tqdm(range(self.n_epochs), position=0, leave=False) as tepoch:
            for epochs in tepoch:
                self.optimizer.zero_grad()
                tepoch.set_description(f"Epoch {epochs}")
                 #for epochs in tqdm(range(self.n_epochs),desc="EPOCHS:",leave=False):
            
                # size = 0
                for (list_in,list_out) in tqdm(zip(list_input,list_output),desc="BATCHES:",leave=False):
                    # print(f"list_in size: {len(list_in)} list_out size: {len(list_out)}")
                    #batch data again

                    input_loader = DataLoader(list_in, batch_size=self.batch, num_workers=0)
                    output_loader = DataLoader(list_out, batch_size=self.batch, num_workers=0)
                    
                    # self.model.optimizer.zero_grad()
                    
                    for list_inin, list_outout in zip(input_loader,output_loader):
                        
                        
                        list_inin = list_inin.to("cuda")
                        list_outout = list_outout.to("cuda")
                        # print(f"\tlist_inin size: {list_inin.shape} list_outout size: {list_outout.shape}")
                        # print(list_inin,list_outout)
                        # print(src_data,tgt_data)
                        # qdecode = datasetss.decode(list_inin)
                        # adecode = datasetss.decode(list_outout)
                        # print(f"Question: {qdecode}\nAnswer{adecode}")
                        output = self.transformer(list_inin, list_outout[:,:-1])

                       
                        
                        # loss = criterion(output.contiguous().view(-1, self.tgt_vocab_size), list_outout[:, 1:].contiguous().view(-1))
                        loss = self.criterion(output.contiguous().view(-1, self.word_size), list_outout[:, 1:].contiguous().view(-1))
                        loss.backward()
                        losses += loss.item()
                        preds = output.argmax(dim=-1)
                        masked_pred = preds * (list_outout[:, 1:]!=0)
                        accuracy = (masked_pred == list_outout[:, 1:]).float().mean()
                        acc += accuracy.item()

                        self.optimizer.step()
                        history_loss.append(loss.item())
                        history_acc.append(accuracy.item())
                        tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy.item())
                        #print(f"Epoch: {epochs+1}, Loss: {loss.item()}")
                model_save_path = "model_checkpoint.pth"
                self.save_model(model_save_path) 


                    
                #     size += len(list_inin)
                # print(f"batch done total size {size}")
            
                    
                    
                    
                    
                    

        

###validation goes here back prob goes heere

if __name__ == "__main__":
    transformers = Transformers()