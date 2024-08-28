###this part for load multiple datasets load secound
from datasets import load_dataset,get_dataset_config_names,get_dataset_split_names,get_dataset_infos
from transformers import GPT2TokenizerFast,BertModel,GPT2Model
from tokenizers import Tokenizer
import numpy as np
import torch
import json
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import h5py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# todo:
#make it handle multiple datasets
#only two file of train test

class Datasets:
    def __init__(self,path='datasets',test_size=0.2) -> None:
        super().__init__()
        
        self.test_size = test_size
        #self.datasets_name = "shivangi19s/kids_chatbot_dataset_4"
        self.splits = 'train'
        # self.batch = batch
        self.token_path = "tokenizer.json"
        self.max_length = 100
        self.path = path
        #set default in-file changed
        self.set_tokenizer = GPT2TokenizerFast(tokenizer_object = Tokenizer.from_file(self.token_path))
         
        self.set_tokenizer.eos_token = self.set_tokenizer.eos_token
        self.set_tokenizer.pad_token = self.set_tokenizer.eos_token
        self.set_tokenizer.bos_token = self.set_tokenizer.bos_token
        
        self.model = GPT2Model.from_pretrained('gpt2').cuda()
        self.model.eval()
        #self.find_datasets = self.Datasets_Finder(self.path)
        


    def datasets_iter(self,datasets,batch):
        print("founded datasets: ",datasets)
        mem_file_path = "file_info.json"
        if os.path.exists(mem_file_path):
            with open(mem_file_path, 'r') as f:
                mem = json.load(f)
        else:
            mem = {"files": {}}
        # store_train = {'train':{}}
        # store_test = {'test':{}}

        for data_path in datasets:
            #save feature
            mem_col = {data_path:[]}

            #directory construct
            store_datasets = {'train': {}, 'test': {}}
            

            #load datasets with split train
            ds = load_dataset('csv',data_files=data_path,split='train')
            #make train/test datasets
            split_datasets = ds.train_test_split(test_size=self.test_size)
            print("Splited datasets info: ",split_datasets)
            

            #extract feature
            features = split_datasets['train'].features
            print("founded feature: ",features)
            #yield train/test
            
            #name for easier naming
            name = os.path.splitext(data_path)[0]
            
            #each feature in each train/test
            for columns in features:
                train_corpus = self.get_train_corpus(split_datasets,batch)
                test_corpus = self.get_test_corpus(split_datasets,batch)
                #store_features = {columns:[]}

                mem_col[data_path].append(columns)
                
                print("operate at label: ",columns)
                #embedded each columns each times appends
                train_embedding = self.embedding(token_path=self.token_path,name=name,datasets=train_corpus,columns=columns,max_length=self.max_length,is_train=True)
                
                store_datasets['train'][columns] = train_embedding

                #embedded each columns each times appends
                test_embedding = self.embedding(token_path=self.token_path,name=name,datasets=test_corpus,columns=columns,max_length=self.max_length,is_train=False)
                
                store_datasets['test'][columns] = test_embedding

            self.save_to_feature(data=store_datasets,file_path=f"{name}_embeddings.feather")
           
            mem['files'].update(mem_col)
            self.save_mem_to_json(mem_file_path,mem)
            #print('test size:',torch.tensor(store_datasets[data_path]['test']['Jarvis']['embeddings']).shape)
            print("hierarchy: ",store_datasets.keys())
            #self.save_to_json(file_path=f"{name}_embeddings.json",data=store_datasets[data_path])
            

            

    def datasets_fetch(self,datasets,batch):
        print("founded datasets: ",datasets)
        mem_file_path = "file_info.json"

        if os.path.exists(mem_file_path):
            with open(mem_file_path, 'r') as f:
                mem = json.load(f)
        else:
            mem = {"files": {}}
            
        # store_train = {'train':{}}
        # store_test = {'test':{}}

        for data_path in datasets:
            #save feature
            mem_col = {data_path:[]}

            #directory construct
            store_datasets = {data_path: {'train': {}, 'test': {}}}
            
            
            #load datasets with split train
            ds = load_dataset('csv',data_files=data_path,split='train')
            #make train/test datasets
            split_datasets = ds.train_test_split(test_size=self.test_size)
            print("Splited datasets info: ",split_datasets)
            
            #extract feature
            features = split_datasets['train'].features
            print("founded feature: ",features)
            
            #name for easier naming
            name = os.path.splitext(data_path)[0]

            train_corpus = self.get_train_corpus(split_datasets,batch)
            test_corpus = self.get_test_corpus(split_datasets,batch)
            #each feature in each train/test
            for columns in features:

                mem_col[data_path].append(columns)

                
                print("operate at label: ",columns)
                #embedded each columns each times appends
                train_embedding = self.embedding(token_path=self.token_path,name=name,datasets=train_corpus,columns=columns,max_length=self.max_length,is_train=True)
        
                store_datasets[data_path]['train'][columns] = train_embedding

                

                mem['files'].update(mem_col)
                self.save_mem_to_json(mem_file_path,mem)

                
                #embedded each columns each times appends
                test_embedding = self.embedding(token_path=self.token_path,name=name,datasets=test_corpus,columns=columns,max_length=self.max_length,is_train=False)
                
                store_datasets[data_path]['test'][columns] = test_embedding

            return store_datasets
                

    
    def Datasets_Finder(self,path):
        datasets_dir_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    datasets_dir_list.append(os.path.join(root, file))
        return datasets_dir_list
    
    def get_train_corpus(self,datasets,batch):
        print("train_datasets size:",len(datasets['train']))
        for i in range(0, len(datasets['train']), batch):
            # print(f"BATCH:{i}")
            batch_data = datasets['train'][i:min(i + batch, len(datasets['train']))]
            #handling left one
            #if len(batch_data) > 0:
            yield batch_data

    def get_test_corpus(self,datasets,batch):
        print("test_datasets size:",len(datasets['test']))
        for i in range(0, len(datasets['test']), batch):
            # print(f"BATCH:{i}")
            batch_data = datasets['test'][i:min(i + batch, len(datasets['test']))]
            #handling left one
            #if len(batch_data) > 0:
            yield batch_data

    def embedding(self,token_path,name,datasets,columns,max_length,is_train):
        
        label = f'{columns}'

        embed_space = {'embeddings':[]}
        save_name = None
        #model = BertModel.from_pretrained('bert-base-uncased')
       
        
       
        #embeddings tokens

        if is_train:
            save_name = "train"
        else:
            save_name = "test"


        rowcount = 0
        for data in datasets:
            
            
            data[label] = [str(v) for v in data[label]]
            
        
            rowcount += len(data[label])

            q_encode = self.set_tokenizer.batch_encode_plus(data[label],padding=True,max_length=max_length,
                                                            truncation=True,add_special_tokens = True,
                                                            return_attention_mask = True, return_tensors='pt')
            
            q_inputs_tensor_id = q_encode['input_ids'].cuda()
            # q_inputs_tensor_mask = q_encode['attention_mask'].cuda()

            # with torch.no_grad(): 
            #         q_outputs = self.model(q_inputs_tensor_id, attention_mask=q_inputs_tensor_mask)

            # last_layer = q_outputs.last_hidden_state.squeeze().tolist()
            # q_embedding = {
            #      'input_ids': q_inputs_tensor_id.squeeze().cpu().numpy().tolist(),
            #      'attention_mask': q_inputs_tensor_mask.squeeze().cpu().numpy().tolist()
            #      #'embeddings': q_outputs.last_hidden_state.squeeze().tolist()
            #  }
            # q_embedding = {
            #     # 'embeddings': q_inputs_tensor_id.squeeze().tolist()
            #     'embeddings':q_outputs.last_hidden_state.squeeze().cpu().numpy().tolist()
            # }
            
            #encode 
            #embed_space['embeddings'].append(q_inputs_tensor_id.tolist())
            embed_space['embeddings'].append(q_inputs_tensor_id)
            print(f"{save_name} {label} concatenated rows: {rowcount} ")
        return embed_space
        
    def save_mem_to_json(self,file_path,data):
        csvList = self.Datasets_Finder("datasets")
        for path in csvList:
            with open(file_path, 'r') as f:
                load = json.load(f)
                if path in load['files']:
                    continue     
                with open(file_path, 'w') as f:
                    #print(f"Open File {file_path} .")
                    print("-----")
                    json.dump(data, f,indent=2)
            print("Close File.")

    # #i did not write this one gpt does //kinda make sense approach of json normalise flatten
    # def flatten(self,data):
    #     flattened_data = {}
    #     for key, value in data.items():
    #         if isinstance(value, dict):
    #             flat = self.flatten(value)
    #             for sub_key, sub_value in flat.items():
    #                 flattened_data[f"{key}_{sub_key}"] = sub_value
    #         else:
    #             flattened_data[key] = value
    #     return flattened_data

    def save_to_json(self,file_path,data):
        with open(file_path, 'w') as f:
            return json.dump(data,f,indent=768)
    
    
    def save_to_feature(self,file_path,data):
        df = pd.DataFrame(data)
        df.to_feather(file_path)



    # def save_to_hdf5(self,file_path,data):
    #     with h5py.File(file_path, 'w') as f:
    #         dset = f.create_dataset('file', data=data, chunks=(1000, 1000), compression='gzip')

    def save_to_parquet(self,file_path,data):
        df = pd.DataFrame(data)
        #df = pd.json_normalize(data, sep='_')
        print("Save Files.")
        table = pa.Table.from_pandas(df)
        pq.write_table(table,file_path)
        #df.to_parquet(file_path,engine='auto',compression='gzip',index=False)


    def decode(self,encode):
        return self.set_tokenizer.batch_decode(encode)
        
if __name__ == "__main__":
    datasets = Datasets()
    #datasets.datasets_iter(datasets=datasets.find_datasets,batch=datasets.batch)