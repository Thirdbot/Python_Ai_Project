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
        self.dropout = 0.4
        self.lr = 0.0001
        self.word_size = 25000
        
        self.n_epochs = 300
        self.batch = 64 #batch in this refer to batch for training

        self.transformer = Transformer(self.src_vocab_size, self.tgt_vocab_size, self.d_model, self.num_heads, self.num_layers, self.d_ff, self.word_size, self.dropout)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)

        self.generator = torch.Generator(device='cuda')

        
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
        
        print(f'\nModel loaded from {path}')
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

    def generate(self,model, src, max_length):
        src = src.to('cuda')
        model.eval()
        end_token = 0
        #tgt = torch.zeros((1, 1)).long().to('cuda')  
        tgt = torch.zeros(src.shape[0], 1, dtype=torch.long).to("cuda")
        cache = [None] * len(model.decoder_layers)
        
        for i in range(max_length):
            output, cache = model(src, tgt[:, -1:], cache=cache)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == end_token:  # assuming 2 is the end token
                break
        
        return tgt
   
    def interference(self):
        file = "model_checkpoint.pth"
        self.transformer = self.load_model(file)
        
        datasetss = Datasets()
        while True:
            sentence = input("You: ")
            if sentence.lower() == "quit":
                break
            
            #print(sentence)
            sentence = sentence.splitlines()
            embbed_sent = self.ListEmbeddings(sentence,100)
            print(embbed_sent)
            embbed_sent = embbed_sent.to("cuda")  # Ensure embeddings are on the correct device
            tgt_data = self.generate(self.transformer,embbed_sent,100)
            #tgt_data = self.top_p_sampling(self.transformer,embbed_sent,100,p=0.5)
            #tgt_data = self.beam_search(self.transformer,embbed_sent,100,100)
            # Decode the generated sequence
            output_tokens = tgt_data.squeeze().tolist()
            decoded_output = datasetss.decode(output_tokens)
            merged_output = self.merge_subword_tokens(decoded_output)
            print("OUTPUT: ",end='')
            for i in merged_output:
                print(i,end='')
            print("\n")
         

    def merge_subword_tokens(self, decoded_sentences):
        """Merge subword tokens in a list of decoded sentences."""
        merged_sentences = []
        for sentence in decoded_sentences:
            words = []
            for word in sentence.split():
                if word.startswith("##"):
                    if words:
                        words[-1] += word[2:]  # Merge with the previous word
                else:
                    words.append(word)
            merged_sentences.append(" ".join(words))
            merged_sentences.append(" ")
        return merged_sentences
    
   

    def ListEmbeddings(self,list_input,word_size):
        datasets_Detail = Datasets()
        embeds = []
        sequence_lengths = []

        embed_input = datasets_Detail.set_tokenizer.batch_encode_plus(list_input, padding='longest',truncation=True,add_special_tokens = True,return_attention_mask = True,max_length=word_size, return_tensors='pt')
        
        #embedded_model = GPT2Model.from_pretrained('gpt2')
        tensor_id = embed_input['input_ids'].long()
        
        tensor_mask = embed_input['attention_mask'].long()
        
        return tensor_id
       


    def batch_sample(self,inpt,outp):
        
        # input_loader = DataLoader(inpt.cpu(), batch_size=self.batch, num_workers=4,shuffle=False,pin_memory=True,generator=self.generator)
        # output_loader = DataLoader(outp.cpu(), batch_size=self.batch, num_workers=4,shuffle=False,pin_memory=True,generator=self.generator)

        # return input_loader,output_loader

        loader = DataLoader(list(zip(inpt.cpu(),outp.cpu())), batch_size=self.batch, num_workers=4,shuffle=False,pin_memory=True,generator=self.generator)
        return loader

    def runtrain(self,list_input,list_output,test_input,test_output):
        if os.path.exists("model_checkpoint.pth"):
            self.transformer = self.load_model(path="model_checkpoint.pth")
        #transformer = nn.DataParallel(self.transformer)
        #optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        datasetss = Datasets()
        
        data = self.batch_sample(list_input,list_output)
        test_data = self.batch_sample(test_input,test_output)

        with tqdm(range(1,self.n_epochs+1), position=0, leave=True) as tepoch:
            # i,o = self.batch_sample(list_input,list_output)

            for epochs in tepoch:
                start_time = time.time()  
                count = 0
                losses = 0
                acc = 0
            
                ####implement dppo for data linked between labels or datasets input model (multimodal)
                
                self.transformer.train()

                with tqdm(data, position=1, leave=True) as tbatch:
                    for list_inin,list_outout in tbatch:
                        
                        count += 1
                        tbatch.set_description(f"Batch step{count}")

                        list_inin = list_inin.cuda()
                        list_outout = list_outout.cuda()

                        # qdecode = datasetss.decode(list_inin[:,:-1])
                        # adecode = datasetss.decode(list_outout[:,:-1])
                        # print(f"\nQuestion: {qdecode}\nAnswer{adecode}\n")
                        output,_ = self.transformer(list_inin, list_outout[:,:-1])
                        #output = self.transformer(list_inin, list_outout)

                    
                        
                        # loss = criterion(output.contiguous().view(-1, self.tgt_vocab_size), list_outout[:, 1:].contiguous().view(-1))
                    
                        loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size), list_outout[:,1:].contiguous().view(-1))

                        loss.backward()

                        losses += loss.item()

                        preds = output.argmax(dim=-1)
                        masked_pred = preds * (list_outout[:,1:]!=0)
                        accuracy = (masked_pred == list_outout[:,1:]).float().mean()
                        acc += accuracy.item()

                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        #tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy.item())

                    #loss,acc of batch element
                    train_loss, train_acc = losses / len(list_outout), acc / len(list_outout)

                    print("\nSize: ",count)

                    end_time = time.time()

                    #loss,acc of whole batch
                    val_loss, val_acc = self.evaluate(self.transformer,list_inin,list_outout, self.criterion)

                    tepoch.set_description(f"Epoch {epochs}")
                    tepoch.set_postfix(trainloss=train_loss, trainaccuracy=train_acc,val_loss=val_loss,val_acc=val_acc)
                    
                   # print((f"\nEpoch: {epochs}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f} "f"Epoch time = {(end_time - start_time):.3f}s\n"))

                    #fine tune whole datasets of batches file
                    # with tqdm(data.dataset, position=2, leave=True) as tune:
                    #     tune.set_description(f"Tune train")
                    #     self.fine_tune(transformer,tune,optimizer=optimizer,criterion=self.criterion,num_epochs=self.n_epochs)

                    # with tqdm(test_data.dataset, position=3, leave=True) as ttune:
                    #     ttune.set_description(f"Tune test")
                    #     self.fine_tune(transformer,ttune,optimizer=self.optimizer,criterion=self.criterion,num_epochs=self.n_epochs)

                model_save_path = "model_checkpoint.pth"
                print("\nsave model\n")
                self.save_model(model_save_path)








    def fine_tune(self,model,data_loader, optimizer, criterion, num_epochs):
        model.train()
        # data_loader = zip(d_in,d_out)
        for epoch in range(num_epochs):
            total_loss = 0
            for src, tgt in data_loader:
                src = src.unsqueeze(0)
                tgt = tgt.unsqueeze(0)
                src = src.cuda()
                tgt = tgt.cuda()
                
                
                self.optimizer.zero_grad()
                
                # # Forward pass
                output, _ = model(src, tgt[:,:-1], cache=None)  # No caching during training
                
                # Compute loss
                loss = criterion(output[:,1:].contiguous().view(-1, self.tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))
                total_loss += loss.item()
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                # #print(f'\nEpoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(tgt)}')


    def evaluate(self,model, inpt,out, loss_fn):
       
        losses = 0
        acc = 0
        
        data = zip(inpt,out)
        with tqdm(data, position=1, leave=False) as tbatch:
            model.eval()
            for x,y in tbatch:
                x = x.cuda()
                y = y.cuda()
                # input_loader = DataLoader(list_in, batch_size=self.batch, num_workers=0,shuffle=True,generator=self.generator)
                # output_loader = DataLoader(list_out, batch_size=self.batch, num_workers=0,shuffle=True,generator=self.generator)

                # for x, y in zip(input_loader,output_loader):
                    
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                logits,_ = model(x, y[:,:-1])

                loss = loss_fn(logits.contiguous().view(-1, self.tgt_vocab_size), y[:,1:].contiguous().view(-1))
                losses += loss.item()
                
                preds = logits.argmax(dim=-1)
                masked_pred = preds * (y[:,1:]!=0)
                accuracy = (masked_pred == y[:,1:]).float().mean()
                acc += accuracy.item()
                

            return losses/len(out) , acc/len(out)


        

###validation goes here back prob goes heere

if __name__ == "__main__":
    transformers = Transformers()