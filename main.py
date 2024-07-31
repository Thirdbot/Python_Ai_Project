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
        
        self.make_file = True
        self.inputs = None
        self.outputs = None

        if(self.CheckNeed(make_file=self.make_file)):
            ###now its time for fetching daatasets each>>>>
            
            couple_list = self.findcouple()
            count = 0
            for data_path in self.file_csv:
                for couple in couple_list[count]:
                    self.inputs = self.soupDatasets(data_path,couple[0],'train',self.make_file)
                    self.outputs = self.soupDatasets(data_path,couple[1],'train',self.make_file)
                    #question-answer pairs this way
                    # print(self.inputs[0])
                    # print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                    # datasets = Datasets()
                    # print(datasets.decode(self.inputs))
                    # print(self.outputs[0])
                    # print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                    # print(datasets.decode(self.outputs))
                    
                count += 1
            ##some model going on here
                        
            ###coupling datasets (may be i not using json file or smt just runtime embeddings osmt)
            pass
        

    #aint test yet
    def soupDatasets(self,data_path,label,type,make_file):
        saved = []
        ##i guess make file work the same dunno dun test yet
        if make_file:
            make_path = "datasets/"+data_path+"_embeddings.json"
            embedd_file = self.load_jsons(make_path)
            for rows in embedd_file[type][label]:
                saved.append(rows['embeddings'])
            return saved
        else:
            make_path = "datasets/"+data_path+".csv"
            embedd_file = self.data_fetch['files'][make_path]
            #print(embedd_file[type].keys())
            for rows in embedd_file[type][label]:
                saved.append(rows['embeddings'])
            return saved

    # def fetchsoup(self,coupling_label,type):
    #     saved = []
    #     i = 0
    #     for data_path in self.file_csv:
    #             make_path = "datasets/"+data_path+".csv"
    #             embedd_file = self.data_fetch['files'][make_path]
    #             for couple in coupling_label[i]:
    #                 i += 1
    #                 for label in couple:
    #                     for rows in embedd_file[type][label]:
    #                         saved.append(rows['embeddings'])
    #     return saved



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
        
            
    # def fetchEmbedd(self,type):
    #     info = self.load_jsons("file_info.json")
    #     for data_path in self.file_csv: 
    #         couple = 2
    #         label_path = "datasets/"+data_path+".csv"
    #         labels = info['files']["datasets/"+data_path+".csv"]
    #         if len(labels) > 2:
    #             for index in range(0,len(labels)-couple):
    #                 #coupling label
    #                 inp = labels[index:index+couple][0]
    #                 out = labels[index:index+couple][1]
    #                 #feed label for outs of its label
    #                 self.inputs = self.fetchsoup(label_path,inp,type)
    #                 self.outputs = self.fetchsoup(label_path,out,type)
    #         elif len(labels) == 2:
                
    #             inp = labels[0]
    #             out = labels[1]
    #             print(inp, " --> " ,out)
    #             #times is size of datasets
    #             self.inputs = self.fetchsoup(label_path,inp,type)
    #             self.outputs = self.fetchsoup(label_path,out,type)
    #         else:
    #             return
        
    
    def load_jsons(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
        
    def CheckNeed(self,make_file):
        File_keep = [file.split("_embeddings")[0] for file in self.file_json]
        datasets = Datasets()
        if make_file:
            for data in self.file_csv:
                split_csv = os.path.splitext(data)[0]
                if split_csv not in File_keep[:][:]:
                    print("making datasetable:")
                    datasets.datasets_iter([f"{self.datapath}/"+f"{data}"+".csv"],self.batch)
                else:
                    continue
        else:
            ##yield function     
            for data in self.file_csv:
                #split_csv = os.path.splitext(data)[0]
                
                path = self.datapath+"/"+data+".csv"
                #print(path)
                #batch
                #C:\Users\astro\Desktop\python_env_project\python_Ai_Project\datasets
                file = datasets.datasets_fetch([path],self.batch)
                #this is how its fetch
                #print(file[path]['train']['Question'][11]['embeddings'])
                self.data_fetch['files'].update(file)
                
                # for data in self.data_fetch:
                #     pass
                    #print(data[path]['train'])
                    ##fetch data save tempo
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