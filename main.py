##for now this part is for load_datasets and tokenize it and more....

import os
from datasets_loader import *
from custom_tokenizer import *

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import GPT2TokenizerFast
import torch
from transformer_model import *
import pandas as pd
import pyarrow.parquet as pq
import dask.dataframe as dd
import dask.array as da
import fastparquet as fp
import test

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device('cuda')

class Program:
    def __init__(self) -> None:
        super(Program).__init__()
        
        self.path = "tokenizer.json"
        try:
           os.path.isfile(self.path)
        except:
            Tokenization()
            self.__init__()
        
        self.datapath = "datasets"
        self.batch = 32 #batch size in this refer to bbatch in save files mean 32 batch for n times
        self.pad_size = 100

        self.data_fetch = {'files':{}}
        self.run_train = True

        self.file_parquet = self.find_datasets(self.datapath,".feather")
        self.file_csv = self.find_datasets(self.datapath,".csv")
        
        #recommend turn to False just to re embeddings it each time(faster than fetch through .json file)
        self.make_file = True

        self.inputs = None
        self.outputs = None

        if(self.CheckNeed(make_file=self.make_file)):
            ###now its time for fetching daatasets each>>>>
            
            couple_list = self.findcouple()
            count = 0
            model = Transformers()
            if (self.run_train):
                for data_path in self.file_csv:
                    for couple in couple_list[count]:

                        self.inputs = self.soupDatasets(data_path,couple[0],'train',self.make_file)
                        self.outputs = self.soupDatasets(data_path,couple[1],'train',self.make_file)
                        #self.embedded(arr=self.inputs)
                        
                        # print(f"run model: {couple}")
                        loss = model.runtrain(self.inputs,self.outputs)

                    count += 1
            if os.path.exists("model_checkpoint.pth"):
                output = model.test_input()
                
        
    def embedded(self,arr):
        embed = nn.Embedding(100,128).to("cuda")
        for a in arr:
            out = embed(a)
        print(out)
    
    def pad_array(self,arr, target_length, padding_value=0):
        sequence_lengths = []
        sequence_lengths.append(arr.shape[0])
        padded_embeds = rnn_utils.pad_sequence(arr, batch_first=False, padding_value=padding_value).to("cuda")
        if padded_embeds.shape[1] < target_length:
            num_padding = target_length - padded_embeds.shape[1]
            padding_tensors = torch.zeros((num_padding, padded_embeds.shape[0])).to("cuda")
            padded_embeds = torch.cat([padded_embeds.transpose(0,1), padding_tensors], dim=0).to("cuda")
            sequence_lengths.extend([0] * num_padding)
        
        return padded_embeds.transpose(0,1).to("cuda")
    
    def pad_encode_array(self,arr, target_length, padding_value=0):
        sequence_lengths = []
        sequence_lengths.append(arr.shape[0])
        #padded_embeds = rnn_utils.pad_sequence(arr, batch_first=False, padding_value=padding_value).to("cuda")
        padded_embeds = arr
        if padded_embeds.shape[0] < target_length:
            num_padding = target_length - padded_embeds.shape[0]
            padding_tensors = torch.zeros((num_padding, padded_embeds.shape[0])).to("cuda")
            padded_embeds = torch.cat([padded_embeds.transpose(-1,0), padding_tensors], dim=0).to("cuda")
            sequence_lengths.extend([0] * num_padding)

        return padded_embeds.transpose(-1,0).to("cuda")
    
    

    #it work
    def soupDatasets(self,data_path,label,type,make_file):
        saved = []
        #Both of these need to change
        if make_file:
            make_path = f"datasets/{data_path}_embeddings.feather"
           
            embedd_file = self.load_feature(make_path)
            
            #make it row by row array
            for stuff in embedd_file[type][label]['embeddings']:
                for row in stuff:
                    numpy_array =torch.stack([torch.tensor(np.array(obj),dtype=torch.long).to("cuda") for obj in row])
                    #padd_arr = self.pad_array(numpy_array,self.pad_size)
                    padd_arr = self.pad_encode_array(numpy_array,self.pad_size)
                    #saved.append(numpy_array)
                    saved.append(padd_arr)
                result = torch.stack(saved).to("cuda")

                yield result.to("cuda")
                #yield torch.tensor(np.array(stuff),dtype=torch.long)
            
            
        
        else:
            make_path = f"datasets/{data_path}.csv"
            embedd_file = self.data_fetch['files'][make_path]
            #print(embedd_file[type].keys())
            for rows in embedd_file[type][label]:
                saved.append(torch.from_numpy(rows['embeddings']))
            return saved

    def findcouple(self):
            info = self.load_jsons("file_info.json")
            store_couple = []
            for data_path in self.file_csv:    
                couple = 2
                #window sliding technique
                labels = info['files']["datasets/"+data_path+".csv"]
                if len(labels) > 2:
                    temp = []
                    for index in range(0,len(labels)-couple):
                        inp = labels[index:index+couple][0]
                        out = labels[index:index+couple][1]
                        temp.append([inp,out])
                    store_couple.append(temp)

                elif len(labels) == 2:
                    inp = labels[0]
                    out = labels[1]
                    store_couple.append([[inp,out]])

                else:
                    return
            return store_couple
        
    # def load_parquet(self,file_path):
    #     parquet_file = pq.ParquetFile(file_path)
    #     df = pq.read_table(file_path,memory_map=True,use_threads=True)
        
    #     # table = pq.read_table(file_path)
    #     # df = table.to_pandas()
    #     # gen = []
    #     # df = pq.ParquetFile(file_path,memory_map=True)
    #     # for row in df.iter_batches(batch_size=10):
    #     #     print('iter...')

    #     #     gen.append(row.to_pandas().to_dict())
    #     # return gen
    #     return df.to_pydict()
    
    def load_feature(self,file_path):
        df = pd.read_feather(file_path)
        return df.to_dict()
    


    # def print_hdf5_structure(self,group, indent=0):
    #     # Print the group name
    #     print('  ' * indent + group.name)
    #     # Iterate over all items in the group
    #     for key, item in group.items():
    #         if isinstance(item, h5py.Group):
    #             # If the item is a group, print its name and recurse
    #             self.print_hdf5_structure(item, indent + 1)
    #         elif isinstance(item, h5py.Dataset):
    #             # If the item is a dataset, print its name
    #             print('  ' * (indent + 1) + key)

    # def load_hdf5(self,file_path):
    #     with h5py.File(file_path, 'r') as f:
    #         self.print_hdf5_structure(f)
    #         dataset = da.from_array(f['file'], chunks=(1000, 1000))
    #         return dataset
        


    def load_jsons(self, file_path):
        with open(file_path, 'rb') as file:
            df = pd.read_json(file)
            return df.to_dict()
            
            
    def CheckNeed(self,make_file):
        File_keep = [file.split("_embeddings")[0] for file in self.file_parquet]
        setsdata = Datasets()
        if make_file:
            for data in self.file_csv:
                split_csv = os.path.splitext(data)[0]
                if split_csv not in File_keep[:][:]:
                    setsdata.datasets_iter([f"{self.datapath}/"+f"{data}"+".csv"],self.batch)
                else:
                    print("passed.")
                    continue
        else:
            ##yield function     
            for data in self.file_csv:
                #split_csv = os.path.splitext(data)[0]
                
                path = self.datapath+"/"+data+".csv"
                #print(path)
                #batch

                file = setsdata.datasets_fetch([path],self.batch)
                #this is how its fetch
                #print(file[path]['train']['Question'][11]['embeddings'])
                self.data_fetch['files'].update(file)
                
        return True


    def find_datasets(self,path,endwith):
        datasets_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(endwith):
                    datasets_list.append(os.path.splitext(file)[0])
        return datasets_list
    
if __name__ == "__main__":
    Program()