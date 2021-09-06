import os
from  pathlib import Path
import shutil
import re
import os




class OBSCCPrepro(object):
    def __init__(self,srcdatapath,dstdatapath,processdir=True):
        self.srcdir=Path(srcdatapath)

                # strip comment from a source file.
        self.NORMALCOMMENT=";[^\r\n]*|#[^\r\n]*|//[^\r\n]*|/\*.*?\*/|'''.*?'''|\"\"\".*?\"\"\""
        self.CSSCOMMENT="//[^\r\n]*|/\*.*?\*/"
        self.OCAMLCOMMENT="\(\*.*?\*\)"

        self.fnMap={"css":self.CSSCOMMENT, "normal":self.NORMALCOMMENT,"OCaml":self.OCAMLCOMMENT}
        if not processdir:
            return 
        self.SOURCETYPE=[]
        for file in self.srcdir.iterdir():
            if os.path.isdir(file):
                self.SOURCETYPE.append(file.name)
        self.dstdatapath=dstdatapath
        self.srcdatapath=srcdatapath
        if Path(dstdatapath).exists():
            shutil.rmtree(dstdatapath)
        Path(dstdatapath).mkdir()



    def StripHTML(self,strSrcCode):

        dr=re.compile("<\s*script\s*[a-z=/\"]*>(.*?)</\s*script\s*>",re.S|re.I)
        so= dr.findall(strSrcCode)
        if so != None:
            return "\n".join(so)
        else:
            dr=re.compile(r'<[^>]*>.*?</[^>]*>|<[^>z]+>|<[\w\s]+>.*?</[\w]+>|<[^<]*?>',re.I|re.S)
            dr=  dr.sub("",strSrcCode)
        return dr

    def StripComment(self,strSrcCode,strType):
        if strType != "" and strType in self.fnMap:
            strPattern=self.fnMap[strType]
        else:
            strPattern=self.fnMap["normal"]

        if strType == "ASP":
            strSrcCode= self.StripHTML(strSrcCode)


    

        pattern1=re.compile(strPattern,re.M|re.I|re.DOTALL)
        result=re.sub(pattern1, '', strSrcCode)
        result=result.replace(r'\r', '')
        result=re.sub(r'\n+', '\n', result)
        result=re.sub(r'^\n+', '', result)
        
        return result


    def StripString(self,strSrcCode,strType):
        dr = re.compile(r"\"\"",re.I)
        strSrcCode=dr.sub("",strSrcCode)
        strPattern="\".*?\"|'.*?'"
        pattern1=re.compile(strPattern,re.I)
        result=re.sub(pattern1, ' STRSTUFF ', strSrcCode)
        return result

    def GetKeyWordSerial(self,strSrcCode):
        strPattern=r"([A-Za-z0-9_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`\"'])"
        pattern1=re.compile(strPattern,re.I)
        result=pattern1.findall(strSrcCode)
        return result

    def ProcessSrcFile(self,src,dst,typename):
        with open(src,encoding="utf-8") as f:
            srccontent=f.read()
            #srccontent=self.StripString(srccontent,typename)
            #srccontent=self.StripComment(srccontent,typename)
            
            with open(dst,mode="w",encoding="utf-8") as nf:
                alllines=self.GetKeyWordSerial(srccontent)
                nf.write("\n".join(alllines))

        

    def DoPreprocess(self):
        for dir in self.SOURCETYPE:
            newdir= Path(self.dstdatapath)/dir
            newdir.mkdir()
            for oldfile in (Path(self.srcdatapath)/dir).iterdir():
            
                newfile=Path(self.dstdatapath)/dir/oldfile.name

                self.ProcessSrcFile(oldfile,newfile,dir)



if __name__ == "__main__" :
    DATASETDIR="data_language_all"
    DATASETCLEAR=DATASETDIR.replace("all","clean")
    DataObj=OBSCCPrepro(DATASETDIR,DATASETCLEAR)
    DataObj.DoPreprocess()

    DATASETDIR="code25_all"
    DATASETCLEAR=DATASETDIR.replace("all","clean")

    DataObj2=OBSCCPrepro(DATASETDIR,DATASETCLEAR)
    DataObj2.DoPreprocess()