###this part for load multiple datasets load secound
from datasets import load_dataset,get_dataset_config_names,get_dataset_split_names,get_dataset_infos
from transformers import GPT2TokenizerFast,BertModel,GPT2Model
from tokenizers import Tokenizer
import numpy as np
import torch
import json
import os


# todo:
#make it handle multiple datasets
#only two file of train test

class Datasets:
    def __init__(self,path='datasets',test_size=0.2,batch=1000) -> None:
        super().__init__()
        
        self.test_size = test_size
        #self.datasets_name = "shivangi19s/kids_chatbot_dataset_4"
        self.splits = 'train'
        self.batch = batch
        self.token_path = "tokenizer.json"
        self.max_length = 100
        self.path = path
        #set default in-file changed
        self.set_tokenizer = GPT2TokenizerFast(tokenizer_object = Tokenizer.from_file(self.token_path))
        self.set_tokenizer.pad_token = self.set_tokenizer.eos_token
        self.set_tokenizer.bos_token = self.set_tokenizer.bos_token
        
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
            store_datasets = {data_path: {'train': {}, 'test': {}}}
            

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
                #store_features = {columns:[]}

                mem_col[data_path].append(columns)

                train_corpus = self.get_train_corpus(split_datasets,batch)
                print("operate at label: ",columns)
                #embedded each columns each times appends
                train_embedding = self.embedding(token_path=self.token_path,name=name,datasets=train_corpus,columns=columns,max_length=self.max_length,is_train=True)
                #should it be this
                #store_datasets[data_path]['train'][columns] = train_embedding[0]
                #store_features[columns].append(train_embedding[0]['embeddings'])
                #store_datasets[data_path]['train'].update(store_features)
                store_datasets[data_path]['train'][columns] = train_embedding

            mem['files'].update(mem_col)
            self.save_mem_to_json(mem_file_path,mem)
            
            self.save_to_json(data=store_datasets[data_path],file_path=f"{name}_embeddings.json")

            #store_datasets[data_path].update(store_train)
            #print(store_datasets[data_path]['train'][0])
            #self.save_to_json(data=store_datasets[data_path],file_path=f"{name}_embeddings.json")

            #for columns in features:
                #store_features = {columns:[]}

            for columns in features:
                test_corpus = self.get_test_corpus(split_datasets,batch)
                print("operate at label: ",columns)
                #embedded each columns each times appends
                test_embedding = self.embedding(token_path=self.token_path,name=name,datasets=test_corpus,columns=columns,max_length=self.max_length,is_train=False)
                #store_datasets[data_path]['test'][columns] = test_embedding[0]
                #store_features[columns].append(test_embedding[0]['embeddings'])
                #store_datasets[data_path]['test'].update(store_features)
                store_datasets[data_path]['test'][columns] = test_embedding

                
            #store_datasets[data_path].update(store_test)
            
            
            self.save_to_json(data=store_datasets[data_path],file_path=f"{name}_embeddings.json")
            

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

            #each feature in each train/test
            for columns in features:

                mem_col[data_path].append(columns)

                train_corpus = self.get_train_corpus(split_datasets,batch)
                print("operate at label: ",columns)
                #embedded each columns each times appends
                train_embedding = self.embedding(token_path=self.token_path,name=name,datasets=train_corpus,columns=columns,max_length=self.max_length,is_train=True)
        
                store_datasets[data_path]['train'][columns] = train_embedding

                

                mem['files'].update(mem_col)
                self.save_mem_to_json(mem_file_path,mem)

            for columns in features:
                test_corpus = self.get_test_corpus(split_datasets,batch)
                print("operate at label: ",columns)
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
            yield datasets['train'][i: i + batch]

    def get_test_corpus(self,datasets,batch):
        print("test_datasets size:",len(datasets['test']))
        for i in range(0, len(datasets['test']), batch):
            yield datasets['test'][i: i + batch]

    def embedding(self,token_path,name,datasets,columns,max_length,is_train):
        
        label = f'{columns}'

        embed_space = []
        save_name = None
        #model = BertModel.from_pretrained('bert-base-uncased')
        model = GPT2Model.from_pretrained('gpt2')
        new_tokenizer = Tokenizer.from_file(token_path)

        self.set_tokenizer = GPT2TokenizerFast(tokenizer_object=new_tokenizer)
        #tokenizer setting
        self.set_tokenizer.pad_token = self.set_tokenizer.eos_token
        self.set_tokenizer.bos_token = self.set_tokenizer.bos_token
        #embeddings tokens

        if is_train:
            save_name = "train"
        else:
            save_name = "test"


        data_get = list(datasets)
        rowcount = 0

        for data in data_get:
            for row in data[label]:
                #alternative is change it to str(datasets) using pandas
                if row == None:
                        row = 'None'

                rowcount += 1
                
                print(f"{save_name} {label}:{rowcount}",row)


                q_encode = self.set_tokenizer.encode_plus(str(row),padding='max_length',max_length=max_length,truncation=True,add_special_tokens = True,return_attention_mask = True, return_tensors='pt')
                
                q_inputs_tensor_id = q_encode['input_ids']
                q_inputs_tensor_mask = q_encode['attention_mask']

                #print(q_encode)
                #print(f"with id: {q_inputs_tensor_id} with size: {len(q_inputs_tensor_id[0])}")
                with torch.no_grad():
                        q_outputs = model(q_inputs_tensor_id, attention_mask=q_inputs_tensor_mask)

                
                # q_embedding = {
                #     'input_ids': q_inputs_tensor_id.squeeze().tolist(),
                #     'attention_mask': q_inputs_tensor_mask.squeeze().tolist(),
                #     'embeddings': q_outputs.last_hidden_state.squeeze().tolist()
                # }
                q_embedding = {
                    #'embeddings': q_inputs_tensor_id.squeeze().tolist()
                    'embeddings': q_outputs.last_hidden_state.squeeze().tolist()
                }
                embed_space.append(q_embedding)
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
    def save_to_json(self,file_path,data):
        with open(file_path, 'w') as f:
            json.dump(data, f,indent=2)
            print("Close File.")

    def decode(self,encode):
        return self.set_tokenizer.batch_decode(encode)
        
if __name__ == "__main__":
    datasets = Datasets()
    #datasets.datasets_iter(datasets=datasets.find_datasets,batch=datasets.batch)