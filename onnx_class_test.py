import onnxruntime as ORT
import onnx 
import pickle as pk
import numpy as np

ONNX_MODEL_PATH="src_cat.onnx"

MAX_TOKEN=400

onnx_model=onnx.load(ONNX_MODEL_PATH)
onnx.checker.check_model(onnx_model)

ort_session = ORT.InferenceSession(ONNX_MODEL_PATH)
print(ORT.get_device())

def DoInference(lines,ort_session):
    ort_inputs={ort_session.get_inputs()[0].name:np.expand_dims(np.array(lines),axis=0).astype(np.int64)}
    ort_outs=ort_session.run(None,ort_inputs)
    return np.argmax(ort_outs[0])
    


def strip_chinese(strs):
    if strs.find("STRSTUFF") > -1 and len(strs)>8:
        print(strs)
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return ""
    return strs

def DoSrcClass(srcfile,ort_session,WORDLIST):
    
    with open(srcfile,"r",encoding="utf-8") as f:
                    lines= f.readlines()  
                    lines=list(map(lambda x:x.replace("\n",""),lines))
                    lines=list(map(lambda x:x.replace("\t",""),lines))
                    lines=list(map(strip_chinese,lines))
                    newlines=[token   for token in lines if token != '' and token !=' ']
                    lines=newlines
                              
                    nLines=len(lines)
                    if  nLines <MAX_TOKEN :
                        lines.extend([""]*(MAX_TOKEN-nLines))
                    else:
                        lines=lines[:MAX_TOKEN]

    newlines=[ WORDLIST[key] for key in lines]
                
    return DoInference(newlines,ort_session)
              



if __name__ == "__main__":
 
    CATDICT={}
    VOCAB={}
    with open("allcat.dat","rb") as f:
        CATDICT=pk.load(f)
    with open("vocab.dat","rb") as f:
        WORDLIST=pk.load(f)
    
    print(WORDLIST["import"])
    catid=DoSrcClass("test.txt",ort_session ,WORDLIST  )

    CATDICT={val:key  for key,val in  CATDICT.items()}
    
   
    print("the class is ",CATDICT[catid])
