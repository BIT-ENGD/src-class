import torch
import torch.nn  as nn
from  torch.utils.data import Dataset,DataLoader
from pathlib import Path
import os

DATASETDIR="data_language_clean"
BATCH_SIZE=128
CLASS_NUM=0
FILE_NUM=0  # 0 means unlimited, otherwise limit to the specifical number.
class BuildSrcData(Dataset):
   
    def __init__(self,DataDir):
        self.allcat={}
        for id,dir in enumerate(Path(DataDir).iterdir()):
            self.allcat[str(dir).split(os.sep)[-1]]=id
        
        for dir in self.allcat:
            for file in (Path(DataDir)/dir).iterdir():
                print(file)
            
        


    def __len__(self):
        return 10

    def __getitem__(self, index):
        return ""
        



class TextCNN(nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super(TextCNN,self).__init__()
        self.W = nn.Embedding(vocab_size,embedding_size)
        output_channel =3
        self.conv = nn.Sequential(
            nn.Conv2d(1,output_channel,(2,embedding_size)),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
        )


def do_train(srcdir):
    SOURCETYPE=[]
    for file in Path(srcdir).iterdir():
        SOURCETYPE.append(file.stem)
    CLASS_NUM=len(SOURCETYPE)
    print(SOURCETYPE)

 







if __name__ == "__main__":
    ds_src=BuildSrcData(DATASETDIR)
    dl=DataLoader(ds_src,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    do_train(DATASETDIR)