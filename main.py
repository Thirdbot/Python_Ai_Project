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

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device('cuda')
print(device)

class Program:
    def __init__(self) -> None:
        super(Program).__init__()
        
        self.path = "tokenizer.json"
        try:
           os.path.isfile(self.path)
        except:
            Tokenization()
            Program()
        
        self.datapath = "datasets"
        self.batch = 1000
        self.data_fetch = {'files':{}}


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
            model = Transformer()
            for data_path in self.file_csv:
                for couple in couple_list[count]:
                    print(couple)

                    self.inputs = self.soupDatasets(data_path,couple[0],'train',self.make_file)
                    self.outputs = self.soupDatasets(data_path,couple[1],'train',self.make_file)
                    torch_inputs = torch.tensor(self.inputs,dtype=torch.float32)
                    torch_outputs = torch.tensor(self.outputs,dtype=torch.float32)
                    print(f"INPUT SHAPE: {torch_inputs.shape}")
                    print(f"OUTPUT SHAPE: {torch_outputs.shape}")

                    print(f"run model: {couple}")
                    model.runtrain(torch_inputs,torch_outputs)
                    
                    # FILE = "data.pth"
                    # data = torch.load(FILE)
                    # model_state = data["model_state"]
                    
                    #question-answer pairs this way
                    #train
                    # print(self.inputs[0])
                    # print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                    # datasets = Datasets()
                    # print(datasets.decode(self.inputs))
                    # print(self.outputs[0])
                    # print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                    # print(datasets.decode(self.outputs))

                    #####IT TIME FOR MODEL
                    
                    
                count += 1
            ##some model going on here
                        
            ###coupling datasets (may be i not using json file or smt just runtime embeddings osmt)
            pass
        

    #it work
    def soupDatasets(self,data_path,label,type,make_file):
        saved = []
        #Both of these need to change
        if make_file:
            make_path = f"datasets/{data_path}_embeddings.feather"
            # print("soup json.")
            
            # file_size = os.path.getsize(make_path)
            # print(f"File size: {file_size / (1024 * 1024)} MB")

            embedd_file = self.load_feature(make_path)

            # print(embedd_file[type][0][label][0]) #for read_table
            #make it row by row array
            for stuff in embedd_file[type][label]['embeddings']:
                for row in stuff:
                    numpy_array = np.array([obj for obj in row], dtype=np.float32)
                    saved.append(numpy_array)
            return np.array(saved)
        
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
        
    def load_parquet(self,file_path):
        parquet_file = pq.ParquetFile(file_path)
        df = pq.read_table(file_path,memory_map=True,use_threads=True)
        
        # table = pq.read_table(file_path)
        # df = table.to_pandas()
        # gen = []
        # df = pq.ParquetFile(file_path,memory_map=True)
        # for row in df.iter_batches(batch_size=10):
        #     print('iter...')

        #     gen.append(row.to_pandas().to_dict())
        # return gen
        return df.to_pydict()
    
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