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

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.set_default_device('cuda')
#simple model embeddings all ready got attention from tokenizer

class Transformer:
    def __init__(self) -> None:
        self.word_size = 100
        self.hiddensize = 1000
        self.ndim = 768
        self.lr = 0.0001
        self.num_layers = 2 #bidirectional
        self.n_epochs = 10
        self.batch = 10
        self.paddings = 100

        enc = Encoder(input_dim=self.ndim,hidden_dim=self.hiddensize,num_layers=self.num_layers)
        dec = Decoder(input_dim=self.ndim,output_dim=self.ndim,hidden_dim=self.hiddensize,num_layers=self.num_layers)

        self.model = Seq2Seq(enc,dec).to('cuda')

    def load_model(self,path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        return self.model
    
    # def test_input(self):
    #     file = "data.pth"
    #     model = self.load_model(file,self.word_size,self.ndim,self.hiddensize)
    #     datasets = Datasets()
    #     classification_layer = torch.nn.Linear(self.ndim, self.ndim)
        
    #     while True:
    #         # sentence = "do you use credit cards?"
    #         sentence = input("You: ")
    #         if sentence == "quit":
    #             break

    #         embbed_sent = self.ListEmbeddings(sentence,self.word_size)
        
    #         with torch.no_grad():
    #             model.eval()
    #             output = model(embbed_sent)
    #             pooled_output = output.mean(dim=1)
    #             logits = classification_layer(pooled_output)
    #              # Get predictions
    #         print("logits: ",logits)
    #         print("logits size: ",logits.shape)
    #         predicted = torch.argmax(logits, dim=-1) 
    #         print("argmax: ",predicted)
    #         print("argmax shape: ",predicted.shape)
    #         # Get probabilities and decode
    #         probs = torch.softmax(logits, dim=1)
    #         print("probs shape: ",probs.shape)
    #         #pred_probs = probs[range(logits.size(0)), predicted]
    #         pred_probs = probs[range(logits.size(0)), predicted]
    #         print("preprobs shape: ",pred_probs.shape)
            
    #         accurate_output = torch.matmul(torch.tensor(pred_probs,dtype=float),torch.tensor(logits,dtype=float))
    #         torch.tensor(accurate_output)
    #         outputs = torch.argmax(accurate_output, dim=-1) 
    #         # Decode and print the prediction
    #         decoded_output = datasets.decode(outputs)
    #         print("Decoded output:", decoded_output)

    def runtrain(self,inputs,outs):
        loss = self.feedmodel(inputs,outs,hiddensize=self.hiddensize,ndim=self.ndim)
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


    def batch_data(self,list_in):
        for i in range(0, len(list_in), self.batch):
            yield list_in[i: i + self.batch]

    def feedmodel(self,list_input,list_output,hiddensize,ndim):
        file = "data.pth"
        
        # if os.path.exists(file):
        #     model = self.load_model(file,self.word_size,self.ndim,self.hiddensize)
        # else:
        #     model = Model(max_length=self.word_size,ndim=ndim,hiddensize=hiddensize,num_layers=self.num_layers,batch=self.batch)
        
        
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_values = []

        #dynamic plot
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        line, = ax.plot([], [], 'r-')
        ax.set_xlim(0, self.n_epochs)
        ax.set_ylim(0, 1)  # Adjust based on expected loss values
        
        for epochs in tqdm(range(self.n_epochs), desc="Training Epochs"):
            self.model.train()
            running_loss = 0.0
            # batch_input = self.batch_data(list_input)
            # batch_output = self.batch_data(list_output)
            
            for list_in, list_out in tqdm(zip(list_input, list_output),desc="Batches"):
                list_in = torch.tensor(list_in, dtype=torch.float32)
                list_in_clone = list_in.clone().requires_grad_(True)
                # print(list_in_clone.shape)
                list_out = torch.tensor(list_out, dtype=torch.float32)
                list_out_clone = list_out.clone().requires_grad_(True)

                predicted = self.model(list_in_clone,list_out_clone)
                
                loss = loss_function(predicted, list_out_clone)
                optimizer.zero_grad()  # Clear gradients before backpropagation
                
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model weights
                running_loss += loss.item()

            epoch_loss = running_loss / len(list_input)
            loss_values.append(epoch_loss)

            # Update the plot
            line.set_xdata(range(len(loss_values)))
            line.set_ydata(loss_values)
            ax.set_ylim(0, max(loss_values) * 1.1)  # Dynamically adjust y-axis
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(list_in_clone.detach(),list_out_clone.detach())
                train_rmse = torch.sqrt(loss_function(y_pred, list_out_clone.detach()))
            print("Epoch %d: train RMSE %.4f" % (epochs, train_rmse))

        

        data = {
            "model_state": self.model.state_dict(),
            "input_size": ndim,
            "hidden_size": hiddensize,
            "output_size": ndim,
            "all_words": self.word_size,
            }

        FILE = "data.pth"
        torch.save(data, FILE)
        plt.ioff()  # Turn off interactive mode
        plt.show()
        print(f'training complete. file saved to {FILE}')
        return loss_values 


# class Model(nn.Module):
#     def __init__(self,max_length,ndim,hiddensize,num_layers,batch) -> None:
#         super(Model,self).__init__()
#         self.max_length = max_length
#         self.ndim = ndim
#         self.hidden_size = hiddensize
#         self.output_size = ndim
#         self.batch = batch
#         self.num_layers = num_layers

#         self.lstm1 = nn.LSTM(self.ndim,self.hidden_size,num_layers=self.num_layers, batch_first=True,bidirectional=True)
#         #comllaspe to linear layer
#         self.last_layer = nn.Linear(2*self.hidden_size,self.output_size)



#     def forward(self,x):
#         batch_size, seq_len, _ = x.size()
#         #memory
#         h_0 = Variable(torch.zeros(2*self.num_layers,batch_size, self.hidden_size).cuda())
#         #carry
#         c_0 = Variable(torch.zeros(2*self.num_layers,batch_size, self.hidden_size).cuda())
#         h_n = h_0.clone()
#         c_n = c_0.clone()

#         outputs = []
#         for t in range(seq_len):
#             # Process the input through LSTM
#             out, (h_n, c_n) = self.lstm1(x[:, t, :].unsqueeze(1), (h_n, c_n))
#             outputs.append(out)

#         outputs = torch.cat(outputs, dim=1)
#         mlp_Out = self.last_layer(outputs)
#         return mlp_Out
    
class Encoder(nn.Module):
    #encode just find feature for decoder by sharing h_0,c_0 in its own encoder then send to decoder
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout,bidirectional=True, batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c_0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim).to(x.device)

        outputs, (hidden, cell) = self.lstm(x,(h_0,c_0))

        return outputs,hidden, cell

class Decoder(nn.Module):
    #decode just translate model from h0,c0 of encoder
    def __init__(self,input_dim, output_dim, hidden_dim, num_layers, dropout=0.5):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        #print("decoder shape: ",predictions.shape)
        return predictions, hidden, cell

#time series seq_seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    #for each src inputs it iterate every target input which mean
    # 100,1,768 -> 100,100,768  1src to 100target then argmax along word_size dim
    #then it loop until src done
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features
        #print("vocab_size",trg_vocab_size)
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(src.device)
        enc_outputs,hidden, cell = self.encoder(src)

        input = trg[:, 0, :].float() 
        #print("Enc shape: ",input.unsqueeze(1).shape)
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)
            outputs[:, t, :] = output.squeeze(1)

            #print("out shape: ",outputs.shape)
            
            #top1 = output.argmax(2)
            #print("top 1 shape",top1.shape)
            input = trg[:, t, :].float() 
           #arg max later

        return outputs


if __name__ == "__main__":
    transformer = Transformer()