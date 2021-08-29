from typing import Optional
import torch
import torch.nn  as nn
from torch.nn.modules.sparse import Embedding
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from pathlib import Path
import os
from  traininfo import TrainInfo
import copy 
from torchsummaryX import summary as summaryx
import onnx 
import pickle as pk
DATASETDIR="data_language_clean"
BATCH_SIZE=256
EMBED_DIM=128
CLASS_NUM=0
FILE_NUM=0  # 0 means unlimited, otherwise limit to the specifical number.
DTYPE=torch.FloatTensor
VOCAB=set()
EPOCH_NUM=500
MAX_TOKEN=400
SEQUENCE_LEN=MAX_TOKEN
FILTER_NUM=3
DROPOUT=0.5
MODEL_NAME="src_cat.pth"
ONNX_MODEL_PATH="src_cat.onnx"

DATASET_FILE="ds.dat"
VOCAB_FILE="vocab.dat"
VOCAB.add("")

def strip_chinese(strs):
    if strs.find("STRSTUFF") > -1 and len(strs)>8:
        print(strs)
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return ""
    return strs

class BuildSrcData(Dataset):
   
    def __init__(self,DataDir,VOCAB):
        self.allcat={}
        self.x_data=[]
        self.y_data=[]
        for id,dir in enumerate(Path(DataDir).iterdir()):
            self.allcat[str(dir).split(os.sep)[-1]]=id
         
        for dir in self.allcat:
            for file in (Path(DataDir)/dir).iterdir():
                with open(file,"r",encoding="utf-8") as f:
                    lines= f.readlines()  
                    lines=list(map(lambda x:x.replace("\n",""),lines))
                    lines=list(map(strip_chinese,lines))

                    newlines=[token   for token in lines if token != '' and token !=' ']
       
                    VOCAB.update(newlines)

                 
                    nLines=len(newlines)
                    if  nLines <MAX_TOKEN :
                        newlines.extend([""]*(MAX_TOKEN-nLines))
                    else:
                        newlines=newlines[:MAX_TOKEN]

                    self.x_data.append(newlines)
                    self.y_data.append(self.allcat[dir])
                    
        
        self.y_data=torch.tensor(self.y_data)

        with open("allcat.dat","wb") as fl:
            pk.dump(self.allcat,fl)
  

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def getnumclass(self):
        return len(self.allcat)
        


#https://wmathor.com/index.php/archives/1445/
'''
class TextCNN(nn.Module):
    def __init__(self,vocab_size,embedding_size,num_classes):
        super(TextCNN,self).__init__()
        self.W = nn.Embedding(vocab_size,embedding_size,padding_idx=0)
        output_channel = 3
        self.conv = nn.Sequential(
            nn.Conv2d(1,output_channel,(2,embedding_size)),
            nn.ReLU(),
            nn.MaxPool2d((2,1)), 
        )

        self.fc=nn.Linear(output_channel,num_classes)
    
    def forward(self,X):
        X=X.transpose(1,0)
        batch_size=X.shape[0]
        embedding_X=self.W(X) # [batch_size, sequence_length, embedding_size]
        embedding_X=torch.unsqueeze(embedding_X,1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv(embedding_X)  # [batch_size, output_channel*1*1]
        flatten = conved.view(batch_size,-1)
        output=self.fc(flatten)
        return output
'''
'''
class TextCNN(nn.Module):
    def __init__(self,vocab_size,embedding_size,num_classes):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(
            # conv : [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
            nn.Conv2d(1, output_channel, (2, embedding_size)),
            nn.ReLU(),
            # pool : ((filter_height, filter_width))
            nn.MaxPool2d((2, 1)),
        )
        # fc
        self.fc = nn.Linear(output_channel, num_classes)

    def forward(self, X):
      
      #X: [batch_size, sequence_length]
      
      batch_size = X.shape[0]
      embedding_X = self.W(X) # [batch_size, sequence_length, embedding_size]
      embedding_X = embedding_X.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
      conved = self.conv(embedding_X) # [batch_size, output_channel, 1, 1]
      flatten = conved.view(batch_size, -1) # [batch_size, output_channel*1*1]
      output = self.fc(flatten)
      return output

'''


class TextCNN(nn.Module):
    def __init__(self,vocab_size,Embedding_size,num_classs):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_dim=Embedding_size)
        out_channel = FILTER_NUM
        self.conv = nn.Sequential(
                    nn.Conv2d(1, out_channel, (2, Embedding_size)),#卷积核大小为2*Embedding_size
                    nn.ReLU(),
                    nn.MaxPool2d((SEQUENCE_LEN-1,1)),
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(out_channel, num_classs)
    
    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X = self.W(X) # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv(embedding_X)# [batch_size, output_channel, 1, 1]
        conved = self.dropout(conved)
        flatten = conved.view(batch_size, -1)# [batch_size, output_channel*1*1]
        output = self.fc(flatten)
        return output

def ExportModel(model,sentence,newmodelpath):
    torch.onnx.export(model,               # model being run
                 sentence,                         # model input (or a tuple for multiple inputs)
                  newmodelpath,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['W'],   # the model's input names
                  output_names = ['fc'], # the model's output names
                  dynamic_axes={'W' : {0:"batch_size"},"fc":{0:"batch_size_output"}}  # variable lenght axes
                                )
    onnx_model=onnx.load(newmodelpath)
    onnx.checker.check_model(onnx_model)
    
def do_train(ds_src,WORDLIST): 
    VOCAB_SIZE=len(WORDLIST) # 
    CLASS_NUM =  ds_src.getnumclass()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = TextCNN(VOCAB_SIZE,EMBED_DIM,CLASS_NUM).to(device)
    print(model)
    criterion   = nn.CrossEntropyLoss().to(device)
    optimizer   = optim.Adam(model.parameters(),lr=5e-4)
    loader      = DataLoader(dataset=ds_src,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    loss        = torch.Tensor([0.0]).float()
    min_loss    = torch.Tensor([10.0]).float().to(device)

# traininfo
    TrainInfoObj=TrainInfo()
  

    input_size = (1,SEQUENCE_LEN)
    x_sample = torch.zeros(input_size, dtype=torch.long, device=torch.device('cuda'))
    print(summaryx(model,x_sample))
    lastsentence=[]
    best_model=model
    for epoch in range(EPOCH_NUM):
        
        for batch_x,batch_y in loader:

            model.train()
            line=[]
            new_batch_x=[]
            for item in batch_x: 
                line=[ WORDLIST[key] for key in item]
                new_batch_x.append(line)

            batch_x=torch.tensor(new_batch_x)
            batch_x=batch_x.transpose(1,0).to(device)
            batch_y=batch_y.to(device)
            pred=model(batch_x)
            loss = criterion(pred,batch_y)
            TrainInfoObj.add_scalar('loss', loss, epoch)
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            if min_loss > loss:
                min_loss =loss
                best_model=copy.deepcopy(model)
                lastsentence=batch_x[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    
  
    if min_loss <10:
        # 
        torch.save(best_model,MODEL_NAME)
        test_sentence=torch.randint(0,46,(1,SEQUENCE_LEN))
        TrainInfoObj.add_graph(model, test_sentence.to(device))
        ExportModel(best_model,test_sentence.to(device),ONNX_MODEL_PATH)
        new_pred=best_model(torch.unsqueeze(lastsentence,0))
        newclass=torch.argmax(new_pred)
        print("ok,newclass:",newclass)


    


if __name__ == "__main__":

    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE,"rb") as f:
           ds_src=pk.load(f)
    else:
        ds_src=BuildSrcData(DATASETDIR,VOCAB)
        with open(DATASET_FILE,"wb") as f:
           pk.dump(ds_src,f)
  
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE,"rb") as f:
            WORDLIST=pk.load(f)
    else:
        WORDLIST={key:i for i,key in enumerate(VOCAB)}
        with open("vocab.dat","wb") as fl:
            pk.dump(WORDLIST,fl)
  
    do_train(ds_src,WORDLIST)
