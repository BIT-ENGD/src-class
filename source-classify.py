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
import torch.nn.functional as F
from  znprompt  import znprompt as zp
from znconfusion  import ConfusionMatrix
DATASETDIR="data_language_clean/"
TRAIN_DIR="train"
VALID_DIR="valid"
BATCH_SIZE=128
EMBED_DIM=128
CLASS_NUM=0
FILE_NUM=0  # 0 means unlimited, otherwise limit to the specifical number.
DTYPE=torch.FloatTensor
VOCAB=set()
EPOCH_NUM=120 #200
MAX_TOKEN=1500
SEQUENCE_LEN=MAX_TOKEN
FILTER_NUM=256
DROPOUT=0.50
MODEL_NAME="src_cat.pth"
ONNX_MODEL_PATH="src_cat.onnx"

DATASET_FILE="ds.dat"
DATASET_VALID_FILE="ds_valid.dat"
VOCAB_FILE="vocab.dat"
CAT_FILE="allcat.dat"
MIN_WORD_FREQUENCE=0 # 3 is good.

VALID_DIR="valid"


def save_var(varname,filename):
    with open(filename,"wb") as f:
        pk.dump(varname,f)

def load_var(filename):
    with open(filename,"rb") as f:
        return pk.load(f)



def strip_chinese(strs):
    if strs.find("STRSTUFF") > -1 and len(strs)>8:
        print(strs)
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return "CNTAGS"
    return strs

class BuildSrcData(Dataset):
   
    def __init__(self,DataDir,VOCAB):
        self.allcat={}
        self.x_data=[]
        self.y_data=[]
        self.vocab_dict={}
        tmp_vocab=set()
        for id,dir in enumerate(Path(DataDir).iterdir()):
            self.allcat[str(dir).split(os.sep)[-1]]=id
         
        for dir in self.allcat:
            for file in (Path(DataDir)/dir).iterdir():
                with open(file,"r",encoding="utf-8") as f:
                    lines= f.readlines()  
                    lines=list(map(lambda x:x.replace("\n",""),lines))
                    lines=list(map(strip_chinese,lines))

                    newlines=[token   for token in lines if token != '' and token !=' ']
       
                    #tmp_vocab.update(newlines)

                             
                    for item in newlines:
                        if item in self.vocab_dict:
                            self.vocab_dict[item]+=1
                        else:
                            self.vocab_dict[item]=1

                    nLines=len(newlines)
                    if  nLines <MAX_TOKEN :
                        newlines.extend([""]*(MAX_TOKEN-nLines))
                    else:
                        newlines=newlines[:MAX_TOKEN]

                    self.x_data.append(newlines)
                    self.y_data.append(self.allcat[dir])
                    
        
        self.y_data=torch.tensor(self.y_data)

        with open(CAT_FILE,"wb") as fl:
            pk.dump(self.allcat,fl)
  

        new_vocab={}
        index=1
        for item in self.vocab_dict:
            if self.vocab_dict[item] >MIN_WORD_FREQUENCE:
                new_vocab[item]=index
                index+=1

        VOCAB.update(new_vocab.keys()) # 赋值
        VOCAB.add("") #添加空字符

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

# cov | max pooling | conv| max pooling | cov | max pooling | full connect | dropout | full connect

class TextCNNEx(nn.Module):
    def __init__(self,vocab_size,Embedding_size,num_classs):
        super(TextCNNEx, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_dim=Embedding_size)
        out_channel = FILTER_NUM
        self.conv1 = nn.Sequential(
                    nn.Conv2d(1, out_channel, (2, Embedding_size)),#卷积核大小为2*Embedding_size　，输入为  [batch,seq_len,embedding]　　，　(Ｗ－Ｆ+２p)/S +1　　，　纵向：　(４００－２－０)/1+1=399　　　，输出 3９９＊１
                    nn.ReLU(),
                    nn.MaxPool2d((SEQUENCE_LEN-1,1)), #  (input +2*p-dilation*(kernelsize-1) -1 )/ stride +1　。 输出为  1*1
        )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(out_channel, out_channel, (2, Embedding_size)),#卷积核大小为2*Embedding_size
                    nn.ReLU(),
                    nn.MaxPool2d((SEQUENCE_LEN-1,1)),
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(out_channel, num_classs)
    
    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X = self.W(X) # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv1(embedding_X)# [batch_size, output_channel, 1, 1]
        #conved = self.conv2(conved)# [batch_size, output_channel, 1, 1]
        conved = self.dropout(conved)
        flatten = conved.view(batch_size, -1)# [batch_size, output_channel*1*1]
        output = self.fc(flatten)
        return output

class textCNN_M(nn.Module):
    def __init__(self, vocab_size,Embedding_size,num_classs):
        super(textCNN_M, self).__init__()
             
        Vocab = vocab_size ## 已知词的数量
        Dim = Embedding_size##每个词向量长度
        Cla = num_classs##类别数
        Ci = 1 ##输入的channel数
        Knum = FILTER_NUM ## 每种卷积核的数量
        Ks = [2,3,5] ## 卷积核list，形如[2,3,4]
        
        self.embed = nn.Embedding(Vocab,Dim) ## 词向量，这里直接随机
        
        self.convs = nn.ModuleList([nn.Conv2d(Ci,Knum,(K,Dim)) for K in Ks]) ## 卷积层
        self.dropout = nn.Dropout(DROPOUT) 
        self.fc = nn.Linear(len(Ks)*Knum,Cla) ##全连接层
        
    def forward(self,x):
        x = self.embed(x) #(N,W,D)
        
        x = x.unsqueeze(1) #(N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line,int(line.size(2))).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        
        x = torch.cat(x,1) #(N,Knum*len(Ks))
        
        x = self.dropout(x)
        logit = self.fc(x)
        return logit


    
def do_train(ds_src,WORDLIST): 
    VOCAB_SIZE=len(WORDLIST) # 
    CLASS_NUM =  ds_src.getnumclass()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = textCNN_M(VOCAB_SIZE,EMBED_DIM,CLASS_NUM).to(device) #TextCNNEx(VOCAB_SIZE,EMBED_DIM,CLASS_NUM).to(device)
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
                line=[ WORDLIST[key] if key  in WORDLIST else 0 for key in item]
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

  # 

def do_valid(WORDLIST,ds_src):

    allcat=load_var(CAT_FILE)
    #print(allcat)
    cmobj=ConfusionMatrix(len(allcat),list(allcat.keys()))
    
    model=torch.load(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    loader      = DataLoader(dataset=ds_src,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    nCorrect=0
    nTotal=0
    for batch_x,batch_y in loader:
        new_batch_x=[]
        for item in batch_x: 
            line=[ WORDLIST[key] if key  in WORDLIST else 0 for key in item]
            new_batch_x.append(line)
        batch_x=torch.tensor(new_batch_x)
        batch_x=batch_x.transpose(1,0).to(device)
        batch_y=batch_y.to(device)
        pred=model(batch_x)
      
        new_pred=torch.argmax(pred,dim=1)
        cmobj.update(new_pred,batch_y)
        result=torch.ones_like(new_pred)*(new_pred==batch_y)
        nCorrect+=torch.sum(result).item()
        nTotal+=new_pred.shape[0]


    cmobj.plot()

    print("valid accuracy: {},total files: {},wrong file:{}".format(nCorrect/nTotal,nTotal,nTotal-nCorrect))

if __name__ == "__main__":

    REFRESHDATA=False
    VALID_ONLY=True
    zpobj=zp()
    if REFRESHDATA :
        if os.path.exists(DATASET_FILE):
            os.remove(DATASET_FILE)
        if os.path.exists(VOCAB_FILE):
            os.remove(VOCAB_FILE)
        if os.path.exists(CAT_FILE):
            os.remove(CAT_FILE)


    if not VALID_ONLY:
        if os.path.exists(DATASET_FILE):
            with open(DATASET_FILE,"rb") as f:
             ds_src=pk.load(f)
        else:
            ds_src=BuildSrcData(DATASETDIR+os.sep+TRAIN_DIR,VOCAB)
            with open(DATASET_FILE,"wb") as f:
             pk.dump(ds_src,f)

    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE,"rb") as f:
            WORDLIST=pk.load(f)
    else:
        WORDLIST={key:i for i,key in enumerate(VOCAB)}
        with open("vocab.dat","wb") as fl:
            pk.dump(WORDLIST,fl)
    if not VALID_ONLY:
        try:
            do_train(ds_src,WORDLIST)
        except:
            zpobj.error()
        else:
            zpobj.finish()
    
    if os.path.exists(DATASET_VALID_FILE):
        with open(DATASET_VALID_FILE,"rb") as f:
            ds_src=pk.load(f)
    else:
        ds_src=BuildSrcData(DATASETDIR+os.sep+VALID_DIR,VOCAB)
        with open(DATASET_VALID_FILE,"wb") as f:
            pk.dump(ds_src,f)
    do_valid(WORDLIST,ds_src)

  



#
