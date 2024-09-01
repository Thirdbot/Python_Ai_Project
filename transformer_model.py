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
        self.src_vocab_size = 52000
        self.tgt_vocab_size = 52000
        self.d_model = 768
        self.num_heads = 16
        self.num_layers = 6
        self.d_ff = 2048
        # max_seq_length = 100
        self.dropout = 0.1
        self.lr = 0.0001
        self.word_size = 52000
        
        self.n_epochs = 100
        self.batch = 8 #batch in this refer to batch for training

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
        
        #tgt = torch.zeros((1, 1)).long().to('cuda')  
        tgt = torch.zeros(src.shape[0], 1, dtype=torch.long).to("cuda")
        cache = [None] * len(model.decoder_layers)
        
        for i in range(max_length):
            output, cache = model(src, tgt[:, -1:], cache=cache)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == 2:  # assuming 2 is the end token
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
            
            embbed_sent = embbed_sent.to("cuda")  # Ensure embeddings are on the correct device
            tgt_data = self.generate(self.transformer,embbed_sent,100)

            # tgt_data = torch.zeros(embbed_sent.shape[0], 1, dtype=torch.long).to("cuda")
            # #tgt_data = torch.zeros((1, 1)).long().to('cuda')
            # cache = [None] * len(self.transformer.decoder_layers)
            # for i in range(1, 100):  # Assuming max length 100
                
            #     with torch.no_grad():
            #         self.transformer.eval()  # Set model to evaluation mode
                    
            #         output,cache  = self.transformer(embbed_sent, tgt_data,cache=cache)

            #         # Get the most likely next token
            #         next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

            #         # Append the predicted token to the target sequence
            #         tgt_data = torch.cat([tgt_data, next_token], dim=1)

            #         # Stop if the model predicts the end token 2 is end token
            #         if next_token.item() == 2:
            #             break

            # Decode the generated sequence
            output_tokens = tgt_data.squeeze().tolist()
            decoded_output = datasetss.decode(output_tokens)
            merged_output = self.merge_subword_tokens(decoded_output)
            print("OUTPUT: ",end='')
            for i in merged_output:
                print(i,end='')
            print("\n")
            # fig = plt.figure()
            # images = self.transformer.decoder_layers[0].cross_attn[0,...].cpu().detach().numpy().mean(axis=0)

            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # ax.set_yticks(range(len(output)))
            # ax.set_xticks(range(len(sentence)))
            # ax.xaxis.set_label_position('top')
            # ax.set_xticklabels(list(sentence))
            # ax.set_yticklabels([f"step {i}" for i in range(len(output))])
            # images = np.clip(images, 0, 1)
            # # images = np.mean(images, axis=0)
            # cax = ax.imshow(images, aspect='auto', cmap='viridis')
            # fig.colorbar(cax)
            # plt.show()  # Ensure the plot is displayed

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

        embed_input = datasets_Detail.set_tokenizer.batch_encode_plus(list_input, padding='max_length',truncation=True,add_special_tokens = True,return_attention_mask = True,max_length=word_size, return_tensors='pt')
        
        #embedded_model = GPT2Model.from_pretrained('gpt2')
        tensor_id = embed_input['input_ids'].long()
        
        tensor_mask = embed_input['attention_mask'].long()
        
        return tensor_id
       


    def batch_sample(self,inpt,outp):
        # datasetss = Datasets()
        # with tqdm(zip(inpt,outp), position=1, leave=False) as tbatch:
            
        #     for list_in,list_out in tbatch:
        input_loader = DataLoader(inpt.cpu(), batch_size=self.batch, num_workers=4,shuffle=False,pin_memory=True,generator=self.generator)
        output_loader = DataLoader(outp.cpu(), batch_size=self.batch, num_workers=4,shuffle=False,pin_memory=True,generator=self.generator)
                # count = 0
                # for list_inin, list_outout in zip(input_loader,output_loader):
                #     count += 1
                #     tbatch.set_description(f"Batch step{count}")
                #     qdecode = datasetss.decode(list_inin[:,:-1])
                #     adecode = datasetss.decode(list_outout[:,:-1])
                #     print(f"\nQuestion: {qdecode}\nAnswer{adecode}")
        return input_loader,output_loader

    def runtrain(self,list_input,list_output):
        # src_data = torch.randint(1, 25000, (1, 100))  # (batch_size, seq_length)
        # tgt_data = torch.randint(1, 25000, (1, 100))  # (batch_size, seq_length)
        
        transformer = nn.DataParallel(self.transformer)

        #load between datasets
        # if os.path.exists("model_checkpoint.pth"):
        #     self.transformer = self.load_model(path="model_checkpoint.pth")
        datasetss = Datasets()
        

        with tqdm(range(1,self.n_epochs+1), position=1, leave=False) as tepoch:
            losses = 0
            acc = 0
            history_loss = []
            history_acc = []
            i,o = self.batch_sample(list_input,list_output)
            for epochs in tepoch:
                start_time = time.time()  
              
                

                ####implement dppo for data linked between labels or datasets input model (multimodal)
                
                transformer.train()
                self.optimizer.zero_grad()
                
                count = 0
                with tqdm(zip(i,o), position=0, leave=True) as tbatch:
                    for list_inin,list_outout in tbatch:
                        count += 1
                        tbatch.set_description(f"Batch step{count}")

                        list_inin = list_inin.cuda()
                        list_outout = list_outout.cuda()

                        # qdecode = datasetss.decode(list_inin[:,:-1])
                        # adecode = datasetss.decode(list_outout[:,:-1])
                        # print(f"\nQuestion: {qdecode}\nAnswer{adecode}\n")
                        output,_ = transformer(list_inin, list_outout[:,:-1],cache=None)
                        #output = self.transformer(list_inin, list_outout)

                    
                        
                        # loss = criterion(output.contiguous().view(-1, self.tgt_vocab_size), list_outout[:, 1:].contiguous().view(-1))
                    
                        loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size), list_outout[:, 1:].contiguous().view(-1))

                        loss.backward()

                        losses += loss.item()

                        preds = output.argmax(dim=-1)
                        masked_pred = preds * (list_outout[:, 1:]!=0)
                        accuracy = (masked_pred == list_outout[:, 1:]).float().mean()
                        acc += accuracy.item()

                        self.optimizer.step()
                        #tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy.item())

                    train_loss, train_acc, hist_loss, hist_acc = losses / len(o), acc / len(o), history_loss, history_acc

                    print("\nSize: ",count)

                    end_time = time.time()

                    val_loss, val_acc, hist_loss, hist_acc = self.evaluate(transformer,i,o, self.criterion)

                    tepoch.set_description(f"Epoch {epochs}")
                    tepoch.set_postfix(trainloss=train_loss, trainaccuracy=train_acc,val_loss=val_loss,val_acc=val_acc)
                    
                    print((f"\nEpoch: {epochs}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f} "f"Epoch time = {(end_time - start_time):.3f}s\n"))

                #self.fine_tune(self.transformer,data_loader=zip(list_in,list_out),optimizer=self.optimizer,criterion=self.criterion,num_epochs=self.n_epochs)

                    model_save_path = "model_checkpoint.pth"
                    print("\nsave model\n")
                    self.save_model(model_save_path)






    # def fine_tune(self,model,data_loader, optimizer, criterion, num_epochs):
    #     model.train()

    #     for epoch in range(num_epochs):
    #         total_loss = 0
    #         for src, tgt in data_loader:
    #             src = src.unsqueeze(0)
    #             tgt = tgt.unsqueeze(0)

    #             self.optimizer.zero_grad()
                
    #             # Forward pass
    #             output, _ = model(src, tgt[:,:-1], cache=None)  # No caching during training
                
    #             # Compute loss
    #             loss = criterion(output.contiguous().view(-1, self.tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))
    #             total_loss += loss.item()
                
    #             # Backward pass and optimization
    #             loss.backward()
    #             optimizer.step()
    #         print(f'\nEpoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(tgt)}')

    def evaluate(self,model, inpt,out, loss_fn):
       
        losses = 0
        acc = 0
        history_loss = []
        history_acc = []
        data = zip(inpt,out)
        with tqdm(data, position=1, leave=False) as tbatch:
            model.eval()
            for x,y in tbatch:
                x = x.cuda()
                y = y.cuda()
                # input_loader = DataLoader(list_in, batch_size=self.batch, num_workers=0,shuffle=True,generator=self.generator)
                # output_loader = DataLoader(list_out, batch_size=self.batch, num_workers=0,shuffle=True,generator=self.generator)

                # for x, y in zip(input_loader,output_loader):
                    
                # x = x.unsqueeze(0)
                # y = y.unsqueeze(0)
                logits,_ = model(x, y[:,:-1])

                loss = loss_fn(logits.contiguous().view(-1, self.tgt_vocab_size), y[:, 1:].contiguous().view(-1))
                losses += loss.item()
                
                preds = logits.argmax(dim=-1)
                masked_pred = preds * (y[:, 1:]!=0)
                accuracy = (masked_pred == y[:, 1:]).float().mean()
                acc += accuracy.item()
                
                history_loss.append(loss.item())
                history_acc.append(accuracy.item())

            return losses/len(inpt.dataset) , acc/len(inpt.dataset) , history_loss, history_acc

                    
                #     size += len(list_inin)
                # print(f"batch done total size {size}")
            
                    
# Credit to Arjun Sarkar and my tweak            
                    
                    
                    

        

###validation goes here back prob goes heere

if __name__ == "__main__":
    transformers = Transformers()