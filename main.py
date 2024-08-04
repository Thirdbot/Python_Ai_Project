##for now this part is for load_datasets and tokenize it and more....

import os
from datasets_loader import *
from custom_tokenizer import *

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import GPT2TokenizerFast
import torch
from transformer_model import *



os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.set_default_device('cuda')
print(torch.get_default_device())

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


        self.file_json = self.find_datasets(self.datapath,".json")
        self.file_csv = self.find_datasets(self.datapath,".csv")
        
        #recommend turn to False just to re embeddings it each time(faster than fetch through .json file)
        self.make_file = False
        self.inputs = None
        self.outputs = None

        if(self.CheckNeed(make_file=self.make_file)):
            ###now its time for fetching daatasets each>>>>
            
            couple_list = self.findcouple()
            count = 0
            model = Transformer()
            for data_path in self.file_csv:
                for couple in couple_list[count]:
                    self.inputs = self.soupDatasets(data_path,couple[0],'train',self.make_file)
                    self.outputs = self.soupDatasets(data_path,couple[1],'train',self.make_file)
                    model.runtrain(self.inputs[0],self.outputs[0])
                    FILE = "data.pth"
                    data = torch.load(FILE)
                    model_state = data["model_state"]
                    
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
        ##i guess make file work the same dunno dun test yet spoiled IT WORK BUT SLOW ASS NOT RECOMMEND TO USE UNLESS YOU HAVE TIME
        if make_file:
            make_path = f"datasets/{data_path}_embeddings.json"
            embedd_file = self.load_jsons(make_path)
            print(embedd_file[type].keys())
            for rows in embedd_file[type][label]:
                saved.append(rows['embeddings'])
            return saved
        else:
            make_path = f"datasets/{data_path}.csv"
            embedd_file = self.data_fetch['files'][make_path]
            #print(embedd_file[type].keys())
            for rows in embedd_file[type][label]:
                saved.append(rows['embeddings'])
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
        
            
    def load_jsons(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"The file {file_path} does not exist.")
            
    def CheckNeed(self,make_file):
        File_keep = [file.split("_embeddings")[0] for file in self.file_json]
        setsdata = Datasets()
        if make_file:
            for data in self.file_csv:
                split_csv = os.path.splitext(data)[0]
                if split_csv not in File_keep[:][:]:
                    print("making datasetable:")
                    setsdata.datasets_iter([f"{self.datapath}/"+f"{data}"+".csv"],self.batch)
                else:
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