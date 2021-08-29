import os
from  pathlib import Path
import shutil
import re


DATASETDIR="data_language_all"

DATASETCLEAR=DATASETDIR.replace("all","clean")

srcdir=Path(DATASETDIR)

SOURCETYPE=[]
for file in srcdir.iterdir():
    SOURCETYPE.append(file.stem)


print(len(SOURCETYPE))


if Path(DATASETCLEAR).exists():
    shutil.rmtree(DATASETCLEAR)


Path(DATASETCLEAR).mkdir()


# strip comment from a source file.
NORMALCOMMENT=";[^\r\n]*|#[^\r\n]*|//[^\r\n]*|/\*.*?\*/|'''.*?'''|\"\"\".*?\"\"\""
CSSCOMMENT="//[^\r\n]*|/\*.*?\*/"
OCAMLCOMMENT="\(\*.*?\*\)"

fnMap={"css": CSSCOMMENT, "normal":NORMALCOMMENT,"OCaml":OCAMLCOMMENT}

def StripHTML(strSrcCode):

    dr=re.compile("<\s*script\s*[a-z=/\"]*>(.*?)</\s*script\s*>",re.S|re.I)
    so= dr.findall(strSrcCode)
    if so != None:
         return "\n".join(so)
    else:
        dr=re.compile(r'<[^>]*>.*?</[^>]*>|<[^>z]+>|<[\w\s]+>.*?</[\w]+>|<[^<]*?>',re.I|re.S)
        dr=  dr.sub("",strSrcCode)
    return dr

def StripComment(strSrcCode,strType,fnMap):
    if strType in fnMap:
        strPattern=fnMap[strType]
    else:
        strPattern=fnMap["normal"]

    if strType == "ASP":
       strSrcCode= StripHTML(strSrcCode)


    

    pattern1=re.compile(strPattern,re.M|re.I|re.DOTALL)
    result=re.sub(pattern1, '', strSrcCode)
    result=result.replace(r'\r', '')
    result=re.sub(r'\n+', '\n', result)
    result=re.sub(r'^\n+', '', result)
    
    return result


def StripString(strSrcCode,strType):
    dr = re.compile(r"\"\"",re.I)
    strSrcCode=dr.sub("",strSrcCode)
    strPattern="\".*?\"|'.*?'"
    pattern1=re.compile(strPattern,re.I)
    result=re.sub(pattern1, ' STRSTUFF ', strSrcCode)
    return result

def GetKeyWordSerial(strSrcCode):
    strPattern=r"([A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`\"'])"
    pattern1=re.compile(strPattern,re.I)
    result=pattern1.findall(strSrcCode)
    return result

def ProcessSrcFile(src,dst,typename,fnMap):
    with open(src,encoding="utf-8") as f:
        if(src.name == "csharp_557.txt"):
            print("csharp")
        srccontent=f.read()
        srccontent=StripString(srccontent,typename)
        srccontent=StripComment(srccontent,typename,fnMap)
        
        with open(dst,mode="w",encoding="utf-8") as nf:
            alllines=GetKeyWordSerial(srccontent)
            nf.write("\n".join(alllines))

        

def DoPreprocess(fnMap):
    for dir in SOURCETYPE:
        newdir= Path(DATASETCLEAR)/dir
        newdir.mkdir()
        for oldfile in (Path(DATASETDIR)/dir).iterdir():
        
            newfile=Path(DATASETCLEAR)/dir/oldfile.name

            ProcessSrcFile(oldfile,newfile,dir,fnMap)



if __name__ == "__main__" :
    DoPreprocess(fnMap)