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
from interference import TrainInterference
import interference


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.set_default_device('cuda')
#simple model embeddings all ready got attention from tokenizer
#alright i gotta change it to gpt model but fast attention



class Transformer:
    def __init__(self) -> None:
        self.word_size = 100
        self.hiddensize = 1000
        self.ndim = 768
        self.lr = 0.000001
        #self.num_layers = 2 #bidirectional
        self.n_epochs = 10
        self.batch = 32 #batch in this refer to batch for training
        self.paddings = 100

        self.train_inter = TrainInterference()
        self.model = self.train_inter.model

    def load_model(self,path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        return self.model
    
    def test_input(self):
        file = "model_checkpoint.pth"
        self.model = self.load_model(file)
        datasets = Datasets()
        classification_layer = torch.nn.Linear(self.ndim, self.ndim)
        
        while True:
            # sentence = "do you use credit cards?"
            sentence = input("You: ")
            if sentence == "quit":
                break

            embbed_sent = self.ListEmbeddings(sentence,self.word_size)
            
            embbed_out = torch.zeros(self.word_size,self.word_size,self.ndim)
            with torch.no_grad():
                self.model.eval()
                output = self.model(embbed_sent,embbed_out)
                pooled_output = output.mean(dim=1)
                logits = classification_layer(pooled_output)
                 # Get predictions
            # print("logits: ",logits)
            # print("logits size: ",logits.shape)
            predicted = torch.argmax(logits, dim=-1) 
            # print("argmax: ",predicted)
            # print("argmax shape: ",predicted.shape)
            # Get probabilities and decode
            probs = torch.softmax(logits, dim=1)
            # print("probs shape: ",probs.shape)
            #pred_probs = probs[range(logits.size(0)), predicted]
            pred_probs = probs[range(logits.size(0)), predicted]
            # print("preprobs shape: ",pred_probs.shape)
            
            accurate_output = torch.matmul(torch.tensor(pred_probs,dtype=float),torch.tensor(logits,dtype=float))
            torch.tensor(accurate_output)
            outputs = torch.argmax(accurate_output, dim=-1) 
            # Decode and print the prediction
            decoded_output = datasets.decode(outputs)
            print("Decoded output:", decoded_output)

    def runtrain(self,inputs,outs):
        loss = self.feedmodel(inputs,outs,hiddensize=self.hiddensize,ndim=self.ndim)
        return loss
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.train_inter.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.word_size = checkpoint.get('word_size', self.word_size)
        self.hiddensize = checkpoint.get('hiddensize', self.hiddensize)
        self.ndim = checkpoint.get('ndim', self.ndim)
        print(f'Model loaded from {path}')
        return self.model
    
    def save_model(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.train_inter.optimizer.state_dict(),
            'word_size': self.word_size,
            'hiddensize': self.hiddensize,
            'ndim': self.ndim,
        }, path)
        print(f'Model saved to {path}')

    def ListEmbeddings(self,list_input,word_size):
        datasets_Detail = Datasets()
        embeds = []
        sequence_lengths = []

        for input in list_input:
            embed_input = datasets_Detail.set_tokenizer.encode_plus(input, padding='max_length',truncation=True,add_special_tokens = True,return_attention_mask = True,max_length=word_size, return_tensors='pt')
            embedded_model = GPT2Model.from_pretrained('gpt2')
            tensor_id = embed_input['input_ids']
            tensor_mask = embed_input['attention_mask']
            with torch.no_grad():
                q_output = embedded_model(tensor_id,attention_mask=tensor_mask)
        
            
            embeds.append(q_output.last_hidden_state.squeeze(0))

        sequence_lengths.append(q_output.last_hidden_state.size(1))

        padded_embeds = rnn_utils.pad_sequence(embeds, batch_first=True, padding_value=0)
        if len(padded_embeds) < self.paddings:
            num_padding = self.paddings - len(padded_embeds)
            padding_tensors = torch.zeros((num_padding, padded_embeds.size(1), padded_embeds.size(2)))
            padded_embeds = torch.cat([padded_embeds, padding_tensors], dim=0)
            sequence_lengths.extend([0] * num_padding)
        return padded_embeds


    # def batch_data(self,list_in,batch):
    #     for i in list_in:
    #         for idx in range(0,len(i),batch):
    #             batched = i[idx:min(idx+batch,len(i))]
    #             yield batched
    
    
    def feedmodel(self,list_input,list_output,hiddensize,ndim):

        #dynamic plot
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        line, = ax.plot([], [], 'r-')
        ax.set_xlim(0, self.n_epochs)
        ax.set_ylim(0, 1)  # Adjust based on expected loss values

        

        # for epochs in tqdm(range(self.n_epochs), desc="Training Epochs"):
        epoch_loss = 0
        self.model.train()
        running_loss = 0.0
        zipdata = zip(list_input,list_output)
        for list_in, list_out in tqdm(zipdata,desc="Batches",leave=False):
            
            #load generator for fast computation
            # inputs_batch = self.batch_data(list_in,batch=self.batch)
            # outputs_batch = self.batch_data(list_out,batch=self.batch)
            #inputs_size = len(list(list_in))
            # print("OUTTER: ",list_in.shape)
            #multithread not really :(
            input_loader = DataLoader(list_in, batch_size=self.batch, num_workers=0,collate_fn=interference.collate_fn)
            output_loader = DataLoader(list_out, batch_size=self.batch, num_workers=0,collate_fn=interference.collate_fn)
            
            self.train_inter.optimizer.zero_grad()
            for list_inin,list_outout in zip(input_loader,output_loader):
                #get each sentence batches may be can get batch token for next tokenprediction instead and treat as senctence??? sliding window techniques by one
                #use fold paddinggs with - infinity in seq_len size for each token slidings
                #iterate sub generator for computertion

                list_inin = list_inin.to("cuda", non_blocking=True)
                list_outout = list_outout.to("cuda", non_blocking=True)

                # self.train_inter.dataloader_train(list_inin)
                # self.train_inter.dataloader_val(list_outout)
                #self.train_inter.dataloader_train(train_itr)
                #d_v = self.train_inter.dataloader_val(val_itr)
                train_itr = self.train_inter.train_iter(list_inin)
                val_itr = self.train_inter.eval_iter(list_outout)
                d_t = self.train_inter.dataloader_train(train_itr)
                d_v = self.train_inter.dataloader_val(val_itr)
                state_loss = self.train_inter.runtrain(d_t,d_v)

                line.set_xdata(range(0,len(state_loss)))
                line.set_ydata(state_loss)
                ax.set_ylim(max(state_loss)*1.1)  
                fig.canvas.draw()
                fig.canvas.flush_events()
                # print("INNER: ",list_inin.shape)
                # predicted = self.model(list_inin,list_outout)
                
                # loss = loss_function(predicted, list_outout)
                
                # loss.backward()  # Compute gradients
                
                # optimizer.step()
                
                # running_loss += loss.item()

                
            # del input_loader
            # del output_loader
            # torch.cuda.empty_cache  
        #     epoch_loss += running_loss / inputs_size
        # loss_values.append(epoch_loss)

        
        #plt.pause(0.01)

        # self.model.eval()
        # for inpt,outp in zipdata:
        #     with torch.no_grad():
        #         y_pred = self.model(inpt.to("cuda", non_blocking=True),outp.to("cuda", non_blocking=True))
                #train_rmse = torch.sqrt(loss_function(y_pred, outp.to("cuda", non_blocking=True)))
    
        plt.ioff()  # Turn off interactive mode
        plt.close()

        model_save_path = "model_checkpoint.pth"
        self.save_model(model_save_path)

    # data = {
    #     "model_state": TrainInterference.model.state_dict()
    #     }
    # FILE = "data.pth"
    # torch.save(data, FILE)
    
    # print(f'training complete. file saved to {FILE}')
    # return loss_values 


# class Encoder(nn.Module):
#     #encode just find feature for decoder by sharing h_0,c_0 in its own encoder then send to decoder
#     def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
#         super(Encoder, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout,bidirectional=True, batch_first=True)
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#     def forward(self, x):
#         batch_size = x.size(0)
#         h_0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim).to(x.device)
#         c_0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim).to(x.device)

#         outputs, (hidden, cell) = self.lstm(x,(h_0,c_0))

#         return outputs,hidden, cell

# class Decoder(nn.Module):
#     #decode just translate model from h0,c0 of encoder
#     def __init__(self,input_dim, output_dim, hidden_dim, num_layers, dropout=0.5):
#         super(Decoder, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(2*hidden_dim, output_dim)

#     def forward(self, x, hidden, cell):
#         outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
#         predictions = self.fc(outputs)
#         #print("decoder shape: ",predictions.shape)
#         return predictions, hidden, cell

# #time series seq_seq
# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     #for each src inputs it iterate every target input which mean
#     # 100,1,768 -> 100,100,768  1src to 100target then argmax along word_size dim
#     #then it loop until src done
#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         batch_size = src.size(0)
#         max_len = trg.size(1)
#         trg_vocab_size = self.decoder.fc.out_features
#         #print("vocab_size",trg_vocab_size)
#         outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(src.device)
#         enc_outputs,hidden, cell = self.encoder(src)

#         input = trg[:, 0, :].float() 
#         #print("Enc shape: ",input.unsqueeze(1).shape)
#         for t in range(1, max_len):
#             output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)
#             outputs[:, t, :] = output.squeeze(1)

#             #print("out shape: ",outputs.shape)
            
#             #top1 = output.argmax(2)
#             #print("top 1 shape",top1.shape)
#             input = trg[:, t, :].float() 
#            #arg max later

#         return outputs


if __name__ == "__main__":
    transformer = Transformer()