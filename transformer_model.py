#this transformer model just make it condition like datasets_loader for use in main file make init and its function
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datasets_loader import *
import os
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.set_default_device('cuda')
#simple model embeddings all ready got attention from tokenizer

class Transformer:
    def __init__(self) -> None:

        self.word_size = 100
        self.hiddensize = 1000
        self.ndim = 768
        self.lr = 0.1
        
        self.n_epochs = 2000
        
        
        # self.runtrain(self.inputsList,self.outputsList)
        # self.test_input()
        
        # for param_tensor in self.model.state_dict():
        #     print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
    
    def test_input(self):
        model = Model(max_length=self.word_size,ndim=self.ndim,hiddensize=self.hiddensize)
        FILE = "data.pth"
        data = torch.load(FILE)
        model_state = data["model_state"]
        model.load_state_dict(model_state)
        model.eval()
        while True:
            # sentence = "do you use credit cards?"
            sentence = input("You: ")
            if sentence == "quit":
                break
            embbed_sent = self.ListEmbeddings(sentence)
            break_embedd = [c for c in embbed_sent]
            datasets = Datasets()
            for emb in break_embedd:
                output = model(emb[0])
                _, predicted = torch.max(output, dim=1)
                
                probs = torch.softmax(output, dim=1)
                prob = probs[0][predicted.item()]
                print(prob)
                print(datasets.decode(predicted))
            

    def runtrain(self,inputs,outs):
        # inp = torch.tensor(inputs, dtype=torch.float32)
        # out = torch.tensor(outs, dtype=torch.float32)
        
        # batchinput = [c for c in inputs]
        # batchoutput = [c for c in outs]
        print(f"inp shape:{inputs.shape} inp shape:{outs.shape}")
        self.feedmodel(inputs,outs,hiddensize=self.hiddensize,ndim=self.ndim)
        
        
            # if prob.item() > 0.75:
            #     for intent in intents['intents']:
            #         if tag == intent["tag"]:
            #             print(f"{bot_name}: {random.choice(intent['responses'])}")
            # else:
            #     print(f"{bot_name}: I do not understand...")

    def ListEmbeddings(self,list_input):
        datasets_Detail = Datasets()
        embeds = []
        for input in list_input:
            embed_input = datasets_Detail.set_tokenizer.encode_plus(input, padding='max_length',truncation=True,add_special_tokens = True,return_attention_mask = True,max_length=self.word_size, return_tensors='pt')
            embedded_model = GPT2Model.from_pretrained('gpt2')
            tensor_id = embed_input['input_ids']
            tensor_mask = embed_input['attention_mask']
            with torch.no_grad():
                q_output = embedded_model(tensor_id,attention_mask=tensor_mask)
        
            q_output.last_hidden_state.squeeze().tolist()
            embeds.append(q_output[0])
        return embeds

    def feedmodel(self,list_input,list_output,hiddensize,ndim):
        model = Model(max_length=self.word_size,ndim=ndim,hiddensize=hiddensize)
        model.eval()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        for epochs in range(self.n_epochs):
            model.train()
            for list_in in list_input:
                predicted = model(list_in)
                for list_out in list_output:               
                    loss = loss_function(predicted,list_out)
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
            if epochs % 100 != 0:
                continue
            model.eval()
            with torch.no_grad():
                y_pred = model(list_in)
                train_rmse = torch.Tensor.cpu(loss_function(y_pred, list_out))
            print("Epoch %d: train RMSE %.4f" % (epochs, train_rmse))
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
        return model


class Model(nn.Module):
    def __init__(self,max_length,ndim,hiddensize) -> None:
        super(Model,self).__init__()
        self.max_length = max_length
        self.ndim = ndim
        self.hidden_size = hiddensize
        self.output_size = ndim
        #self.batch = 10
        # 2 layer lstm translate layer
        self.lstm1 = nn.LSTM(self.ndim,self.hidden_size,batch_first=True)
        
        #memory
        self.h_0 = Variable(torch.zeros(1, self.hidden_size).cuda())
        #carry
        self.c_0 = Variable(torch.zeros(1, self.hidden_size).cuda())

        #comllaspe to linear layer
        self.last_layer = nn.Linear(self.hidden_size,self.output_size)

        

        #each word got its memory and cary throughout hidddensize so it hiddensize//2*inputshape[1] worth of word memory
        

        
    #translate wordx768 to 768//word features worth of h_0 and c_0 which is size of wordx768 and it forward
    #7x768 of history word then it repleate itself for word-1 times per word_senctence ##slide window techniques
    ##that is one encode layers

    def forward(self,x):
        
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