import os
from  pathlib import Path
import shutil
import re


DATASETDIR="data_language_all"

DATASETCLEAR=DATASETDIR.replace("all","clear")

srcdir=Path(DATASETDIR)

SOURCETYPE=[]
for file in srcdir.iterdir():
    SOURCETYPE.append(file.stem)


print(len(SOURCETYPE))


if Path(DATASETCLEAR).exists():
    shutil.rmtree(DATASETCLEAR)


Path(DATASETCLEAR).mkdir()


NORMALCOMMENT=";[^\r\n]*|#[^\r\n]*|//[^\r\n]*|/\*.*?\*/|'''.*?'''|\"\"\".*?\"\"\""
CSSCOMMENT="//[^\r\n]*|/\*.*?\*/"

fnMap={"css": CSSCOMMENT, "normal":NORMALCOMMENT}

def StripComment(strSrcCode,strType,fnMap):
    if strType in fnMap:
        strPattern=fnMap[strType]
    else:
        strPattern=fnMap["normal"]

    pattern1=re.compile(strPattern,re.M|re.I|re.DOTALL)
    result=re.sub(pattern1, '', strSrcCode)
    result=result.replace(r'\r', '')
    result=re.sub(r'\n+', '\n', result)
    
    return result


def GetKeyWordSerial(strSrcCode):

    return strSrcCode

def ProcessSrcFile(src,dst,typename):
    global fnMap
    with open(src,encoding="utf-8") as f:
        srccontent=f.read()
        srccontent=StripComment(srccontent,typename,fnMap)
        with open(dst,mode="w",encoding="utf-8") as nf:
            alllines=GetKeyWordSerial(srccontent)
            nf.write(alllines)

        

def DoPreprocess():
    for dir in SOURCETYPE:
        newdir= Path(DATASETCLEAR)/dir
        newdir.mkdir()
        for oldfile in (Path(DATASETDIR)/dir).iterdir():
        
            newfile=Path(DATASETCLEAR)/dir/oldfile.name

            ProcessSrcFile(oldfile,newfile,dir)



if __name__ == "__main__" :
    DoPreprocess()