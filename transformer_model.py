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
import test


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.set_default_device('cuda')
#simple model embeddings all ready got attention from tokenizer
#alright i gotta change it to gpt model but fast attention



class Transformers:
    def __init__(self) -> None:
        self.word_size = 100
        self.hiddensize = 256
        self.ndim = 768
        self.lr = 0.000001
        #self.num_layers = 2 #bidirectional
        self.n_epochs = 10
        self.batch = 16 #batch in this refer to batch for training
        self.paddings = 100

        # self.train_inter = TrainInterference()
        
        self.args = {
            'vocab_size': 100,
            'model_dim': 768,
            'dropout': 0.1,
            'n_encoder_layers': 1,
            'n_decoder_layers': 1,
            'n_heads': 4
        }
        self.model = test.Test_Model(self.args)
        self.transformer = self.model.model


    def load_model(self, path):
        checkpoint = torch.load(path)
        
        # Load model state
        self.transformer.load_state_dict(checkpoint['model_state'])
        
        # Optionally load optimizer state if available
        if 'optimizer_state' in checkpoint:
            self.model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Update internal attributes
        self.word_size = checkpoint.get('word_size', self.word_size)
        self.hiddensize = checkpoint.get('hiddensize', self.hiddensize)
        self.ndim = checkpoint.get('ndim', self.ndim)
        
        print(f'Model loaded from {path}')
        return self.transformer

    def save_model(self, path):
        # Save model state
        torch.save({
            'model_state': self.transformer.state_dict(),
            'optimizer_state': self.model.optimizer.state_dict() if hasattr(self.model, 'optimizer') else None,
            'word_size': self.word_size,
            'hiddensize': self.hiddensize,
            'ndim': self.ndim,
        }, path)
        print(f'Model saved to {path}')

    def test_input(self):
        file = "model_checkpoint.pth"
        self.transformer = self.load_model(file)
        self.transformer.eval()  # Set model to evaluation mode
        
        while True:
            sentence = input("You: ")
            if sentence.lower() == "quit":
                break

            embbed_sent = self.ListEmbeddings(sentence, self.word_size)
            embbed_sent = embbed_sent.to("cuda")  # Ensure embeddings are on the correct device

            with torch.no_grad():
                # Assuming Translator is a method/function of the model
                output = test.Translator(self.transformer)(embbed_sent, max_length=self.paddings)
                print(output)

            
    def runtrain(self,inputs,outs):
        loss = self.feedmodel(inputs,outs,hiddensize=self.hiddensize,ndim=self.ndim)
        return loss

    def ListEmbeddings(self,list_input,word_size):
        datasets_Detail = Datasets()
        embeds = []
        sequence_lengths = []

        for input in list_input:
            embed_input = datasets_Detail.set_tokenizer.encode_plus(input, padding='longest',truncation=True,add_special_tokens = True,return_attention_mask = True,max_length=word_size, return_tensors='pt')
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
        else:
            padded_embeds = padded_embeds.transpose(0,1).to("cuda")
        return padded_embeds.transpose(0,1)



    
    def feedmodel(self,list_input,list_output,hiddensize,ndim):

        # #dynamic plot
        # plt.ion()
        # fig, ax = plt.subplots()
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Loss')
        # line, = ax.plot([], [], 'r-')
        # ax.set_xlim(0, self.n_epochs)
        # ax.set_ylim(0, 1)  # Adjust based on expected loss values

        self.transformer.train()
        zipdata = zip(list_input,list_output)

        for epochs in tqdm(range(self.n_epochs),leave=False):

            for list_in, list_out in zipdata:
                
                #batch data again
                input_loader = DataLoader(list_in, batch_size=self.batch, num_workers=4)
                output_loader = DataLoader(list_out, batch_size=self.batch, num_workers=4)
                
                self.model.optimizer.zero_grad()
                zipdata2 = zip(input_loader,output_loader)
                for list_inin,list_outout in zipdata2:
                    print(list_inin.shape)
                    list_inin = list_inin.to("cuda", non_blocking=True)
                    list_outout = list_outout.to("cuda", non_blocking=True)
                    print("INPUT SHAPE: ",list_inin.shape)
                    print("OUTPUT SHAPE: ",list_outout.shape)
                    
                    self.model.setup(list_inin,list_outout)
                    self.model.run()
                    
                    
            # plt.ioff()  # Turn off interactive mode
            # plt.close()

        model_save_path = "model_checkpoint.pth"
        self.save_model(model_save_path)



if __name__ == "__main__":
    transformers = Transformers()