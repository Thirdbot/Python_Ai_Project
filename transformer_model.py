#this transformer model just make it condition like datasets_loader for use in main file make init and its function
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datasets_loader import *
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.set_default_device('cuda')
#simple model embeddings all ready got attention from tokenizer

class Transformer:
    def __init__(self) -> None:
        self.word_size = 25000

        self.input = "hello, my name is Third."
        self.output = "hello, my name is Bot."

        self.batchinput = [c for c in self.input]
        self.batchoutput = [c for c in self.output]
        

        self.inputsList = self.ListEmbeddings(self.batchinput)
        self.outputsList = self.ListEmbeddings(self.batchoutput)
        self.model = self.feedmodel(self.inputsList[0],self.outputsList[0])
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        
        
    def ListEmbeddings(self,list_input):
        datasets_Detail = Datasets()
        embeds = []
        for input in list_input:
            embed_input = datasets_Detail.set_tokenizer.encode_plus(input, padding=True,add_special_tokens = True,truncation=True,return_attention_mask = True,max_length=self.word_size, return_tensors='pt')
            embedded_model = GPT2Model.from_pretrained('gpt2')
            tensor_id = embed_input['input_ids']
            tensor_mask = embed_input['attention_mask']
            with torch.no_grad():
                q_output = embedded_model(tensor_id,attention_mask=tensor_mask)
        
            q_output.last_hidden_state.squeeze().tolist()
            embeds.append(q_output[0])
        return embeds

    def feedmodel(self,list_input,list_output):
        model = Model(max_length=self.word_size,ndim=768,hiddensize=1000)
        for list_in in list_input:
            fn = model.forward(list_in)
            model.backward(model,fn,list_output[0])
        return model


class Model(nn.Module):
    def __init__(self,max_length,ndim,hiddensize) -> None:
        super(Model,self).__init__()
        self.max_length = max_length
        self.ndim = ndim
        self.hidden_size = hiddensize
        self.output_size = ndim

        # 2 layer lstm translate layer
        self.lstm1 = nn.LSTM(self.ndim,self.hidden_size)
        
        #comllaspe to linear layer
        self.last_layer = nn.Linear(self.hidden_size,self.output_size)

        #each word got its memory and cary throughout hidddensize so it hiddensize//2*inputshape[1] worth of word memory
        

        
    #translate wordx768 to 768//word features worth of h_0 and c_0 which is size of wordx768 and it forward
    #7x768 of history word then it repleate itself for word-1 times per word_senctence ##slide window techniques
    ##that is one encode layers

    def forward(self,x):
        #memory
        h_0 = Variable(torch.zeros(1, self.hidden_size).cuda())
        #carry
        c_0 = Variable(torch.zeros(1, self.hidden_size).cuda())
        
        output1, (final_hidden_state1, final_cell_state) = self.lstm1(x, (h_0, c_0))
        mlp_Out = self.last_layer(output1)
        output2, (final_hidden_state2, final_cell_state) = self.lstm1(mlp_Out, (final_hidden_state1, final_cell_state))
        h_0,c_0 = final_hidden_state2,final_cell_state

        print("OUTPUT: ",output2)
        print("OUTPUT SIZE: ",output2.shape)

        #self.input = output

        
        output = self.last_layer(output2) 
        output = F.log_softmax(output, dim=1)
        print("SOFTMAX1: ",output)
        return output
    
    def backward(self,model,prob_out,out):
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model.zero_grad()
        loss = loss_function(prob_out,out)
        loss.backward()
        optimizer.step()
        

if __name__ == "__main__":
    transformer = Transformer()