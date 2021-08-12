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


def StripComment(strSrcCode):

    return strSrcCode


def GetKeyWordSerial(strSrcCode):

    return strSrcCode

def ProcessSrcFile(src,dst):
    with open(src,encoding="utf-8") as f:
        srccontent=f.read()
        srccontent=StripComment(srccontent)
        with open(dst,mode="w",encoding="utf-8") as nf:
            alllines=GetKeyWordSerial(srccontent)
            nf.write(alllines)

        

def DoPreprocess():
    for dir in SOURCETYPE:
        newdir= Path(DATASETCLEAR)/dir
        newdir.mkdir()
        for oldfile in (Path(DATASETDIR)/dir).iterdir():
        
            newfile=Path(DATASETCLEAR)/dir/oldfile.name

            ProcessSrcFile(oldfile,newfile)



if __name__ == "__main__" :
    DoPreprocess()