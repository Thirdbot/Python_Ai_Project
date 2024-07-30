##for now this part is for load_datasets and tokenize it and more....

import os
from datasets_loader import *
from custom_tokenizer import *

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import GPT2TokenizerFast
import torch


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.set_default_device('cuda')

print(torch.cuda.is_available(),torch.cuda.device)
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

        self.file_json = self.find_datasets(self.datapath,".json")
        self.file_csv = self.find_datasets(self.datapath,".csv")

        if(self.CheckNeed(make_file=False)):
            ###now its time for fetching daatasets each>>>>
            INP,OUT = self.FetchDatasets('train')
            ###coupling datasets (may be i not using json file or smt just runtime embeddings osmt)
            pass
        

    #aint test yet
    def SoupDatasets(self,path,label,type,times):
        for embed in self.getEmbedd(path,label,type,times):
            print(embed)

    #aint test yet
    def getEmbedd(self,path,label,type,time):
        make_path = "datasets\\"+path+"_embeddings.json"
        file_data = self.load_jsons(make_path)
        i = 0
        for i in range(time):
            yield file_data[type][label][i]['embeddings']
            i += 1
            
    
    def FetchDatasets(self,type):
        #loop each datasets
        info = self.load_jsons("file_info.json")
        for data_path in self.file_csv:
            
            couple = 2
            #window sliding technique
            labels = info['files']["datasets/"+data_path+".csv"]
            if len(labels) > 2:
                for index in range(0,len(labels)-couple):
                    inp = labels[index:index+couple][0]
                    out = labels[index:index+couple][1]
                    print(inp, " --> " ,out)
                    #times is size of datasets
                    self.SoupDatasets(data_path,inp,type,10)
                    self.SoupDatasets(data_path,out,type,10)
            elif len(labels) == 2:
                inp = labels[0]
                out = labels[1]
                print(inp, " --> " ,out)
                #times is size of datasets
                self.SoupDatasets(data_path,inp,type,10)
                self.SoupDatasets(data_path,out,type,10)
            else:
                return
            
    def load_jsons(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
        
    def CheckNeed(self,make_file):
        File_keep = [file.split("_embeddings")[0] for file in self.file_json]
        if make_file:
            for data in self.file_csv:
                split_csv = os.path.splitext(data)[0]
                if split_csv not in File_keep[:][:]:
                    datasets = Datasets()
                    datasets.datasets_iter([f"{self.datapath}//"+f"{data}"+".csv"],self.batch)
                else:
                    continue
        else:
            ##yield function     
            for data in self.file_csv:
                #split_csv = os.path.splitext(data)[0]
                datasets = Datasets()
                path = self.datapath+"\\"+data+".csv"
                print(path)
                #batch
                #C:\Users\astro\Desktop\python_env_project\python_Ai_Project\datasets
                data_fetch = datasets.datasets_fetch(path,self.batch)
                print(next(data_fetch))
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