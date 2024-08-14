#this transformer model just make it condition like datasets_loader for use in main file make init and its function
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datasets_loader import *
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.set_default_device('cuda')
#simple model embeddings all ready got attention from tokenizer

class Transformer:
    def __init__(self) -> None:
        self.word_size = 100
        self.hiddensize = 1000
        self.ndim = 768
        self.lr = 0.001
        self.num_layers = 2 #bidirectional ####datasets_size > num_layer *batch
        self.n_epochs = 10
        self.batch = 30
        
        # self.runtrain(self.inputsList,self.outputsList)
        # self.test_input()
        
        # for param_tensor in self.model.state_dict():
        #     print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
    def load_model(self,path, word_size, ndim, hiddensize):
        model = Model(max_length=word_size, ndim=ndim, hiddensize=hiddensize,num_layers=self.num_layers,batch=self.batch)
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])
        return model
    
    def test_input(self):
        file = "data.pth"
        model = self.load_model(file,self.word_size,self.ndim,self.hiddensize)
        datasets = Datasets()
        classification_layer = torch.nn.Linear(self.ndim, self.word_size)
        
        while True:
            # sentence = "do you use credit cards?"
            sentence = input("You: ")
            if sentence == "quit":
                break

            embbed_sent = self.ListEmbeddings(sentence,self.word_size)
            #emb_tensor = torch.tensor(embbed_sent, dtype=torch.float32)
            # torch.tensor(embbed_sent)
            # # break_embedd = [c for c in embbed_sent]s
            # emb_tensor = embbed_sent.clone().detach()
            # emb_tensor = emb_tensor.to(next(model.parameters()).device)
            
            with torch.no_grad():
                model.eval()
                output = model(embbed_sent)
                pooled_output = output.mean(dim=1)
                logits = classification_layer(pooled_output)
                 # Get predictions
            print(logits)
            predicted = torch.argmax(logits, dim=-1) 
            
            # Get probabilities and decode
            probs = torch.softmax(logits, dim=1)
            pred_probs = probs[range(logits.size(0)), predicted]
            for i, prob in enumerate(pred_probs):
                print(f"Probability of predicted class for sample {i}: {prob.item()}")
            
            # Decode and print the prediction
            decoded_output = datasets.decode(predicted)
            print("Decoded output:", decoded_output)

    def runtrain(self,inputs,outs):
        # print(f"inp shape:{inputs.shape} inp shape:{outs.shape}")

        self.feedmodel(inputs,outs,hiddensize=self.hiddensize,ndim=self.ndim)
        
        
            # if prob.item() > 0.75:
            #     for intent in intents['intents']:
            #         if tag == intent["tag"]:
            #             print(f"{bot_name}: {random.choice(intent['responses'])}")
            # else:
            #     print(f"{bot_name}: I do not understand...")

    def ListEmbeddings(self,list_input,word_size):
        datasets_Detail = Datasets()
        embeds = []
        sequence_lengths = []
        #sequence_lengths.append(self.batch)

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
        if len(padded_embeds) < self.batch:
            num_padding = self.batch - len(padded_embeds)
            padding_tensors = torch.zeros((num_padding, padded_embeds.size(1), padded_embeds.size(2)))
            padded_embeds = torch.cat([padded_embeds, padding_tensors], dim=0)
            sequence_lengths.extend([0] * num_padding)
        return padded_embeds


    def batch_data(self,list_in):
        for i in range(0, len(list_in), self.batch):
            yield list_in[i: i + self.batch]

    def feedmodel(self,list_input,list_output,hiddensize,ndim):
        
        model = Model(max_length=self.word_size,ndim=ndim,hiddensize=hiddensize,num_layers=self.num_layers,batch=self.batch)
        
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_values = []

       
        for epochs in range(self.n_epochs):
            model.train()
            running_loss = 0.0
            batch_input = self.batch_data(list_input)
            batch_output = self.batch_data(list_output)
        
            # for list_in in list_input:
            #     predicted = model(list_in)
            #     for list_out in list_output:               
            #         loss = loss_function(predicted,list_out)
            #         optimizer.zero_grad()
            #         loss.backward(retain_graph=True)
            #         optimizer.step()
            for list_in, list_out in zip(batch_input, batch_output):
                #list_in = torch.tensor(list_in, dtype=torch.float32)
                list_in_clone = list_in.clone().detach().requires_grad_(True)
                #list_out = torch.tensor(list_out, dtype=torch.float32)
                list_out_clone = list_out.clone().detach().requires_grad_(True)
                predicted = model(list_in_clone)
                
                loss = loss_function(predicted, list_out_clone)
                optimizer.zero_grad()  # Clear gradients before backpropagation
                
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model weights
                running_loss += loss.item()

                epoch_loss = running_loss / len(list_input)
                loss_values.append(epoch_loss)

            #if epochs % 1 == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(list_in_clone.detach())
                train_rmse = torch.Tensor.cpu(loss_function(y_pred, list_out_clone.detach()))
            print("Epoch %d: train RMSE %.4f" % (epochs, train_rmse))

        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()

        data = {
            "model_state": model.state_dict(),
            "input_size": ndim,
            "hidden_size": hiddensize,
            "output_size": ndim,
            "all_words": self.word_size,
            }

        FILE = "data.pth"
        torch.save(data, FILE)

        print(f'training complete. file saved to {FILE}')
        return loss_values 


class Model(nn.Module):
    def __init__(self,max_length,ndim,hiddensize,num_layers,batch) -> None:
        super(Model,self).__init__()
        self.max_length = max_length
        self.ndim = ndim
        self.hidden_size = hiddensize
        self.output_size = ndim
        self.batch = batch
        self.num_layers = num_layers
        # 2 layer lstm translate layer
        self.lstm1 = nn.LSTM(self.ndim,self.hidden_size,num_layers=self.num_layers, batch_first=True,bidirectional=True)
        
        #memory
        self.h_0 = Variable(torch.zeros(2*self.num_layers,self.batch, self.hidden_size).cuda())
        #carry
        self.c_0 = Variable(torch.zeros(2*self.num_layers,self.batch, self.hidden_size).cuda())

        #comllaspe to linear layer
        self.last_layer = nn.Linear(2*self.hidden_size,self.output_size)

        

        #each word got its memory and cary throughout hidddensize so it hiddensize//2*inputshape[1] worth of word memory
        

        
    #translate wordx768 to 768//word features worth of h_0 and c_0 which is size of wordx768 and it forward
    #7x768 of history word then it repleate itself for word-1 times per word_senctence ##slide window techniques
    ##that is one encode layers

    def forward(self,x):
        
        # padd_seq = pad_packed_sequence(x,batch_first=True)
        # lengths = torch.tensor([len(seq) for seq in x])
        # packed_input = pack_padded_sequence(padd_seq, lengths, batch_first=True, enforce_sorted=False)

        output1, (final_hidden_state1, final_cell_state1) = self.lstm1(x, (self.h_0, self.c_0))
        #self.h_0,self.c_0 = final_hidden_state2,final_cell_state
        mlp_Out = self.last_layer(output1)
        output2, (final_hidden_state2, final_cell_state2) = self.lstm1(mlp_Out, (final_hidden_state1, final_cell_state1))

        #self.input = output

        
        output = self.last_layer(output2) 
        # output = F.log_softmax(output, dim=1)
        # print("SOFTMAX1: ",output)
        return output
    

if __name__ == "__main__":
    transformer = Transformer()