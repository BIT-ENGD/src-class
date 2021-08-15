
import re

'''
// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.



'''


str='''
for( int i =0; i<5; i++)
{

//中国人好样的，不是吗？
}

do
{

printf("ahahah"); // 真的行？

}while(i>5);

//中国人好样
// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#中国人好样的
/* 
花木成畦手自栽 
花木成畦手自栽
asdfdadfs 
*/

\'\'\'
asdf的发射点大师傅ｄ
\'\'\'
'''
#pattern1=re.compile("#[^\r\n]*|//[^\r\n]*|/\*.*?\*/|'''.*?'''",re.M|re.I|re.DOTALL)
strPattern="[\w]+|[\"\"!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~\"\"\]"
pattern1=re.compile(strPattern,re.I)
#result=pattern1.findall("if (list_empty(&ubi->works)) {")
#print(result)


def StripHTML(strSrcCode):
    #dr = re.compile(r"<[^>]+>|<[\w\s]+>.*?</[\w]+>|<[^<]*?>",re.I)
    dr=re.compile("<\s*script\s*[a-z=/\"]*>(.*?)</\s*script\s*>",re.S|re.I)
    so= dr.findall(strSrcCode)
    if so != None:
         return "\n".join(so)
    else:
        dr=re.compile(r'<[^>]*>.*?</[^>]*>|<[^>z]+>|<[\w\s]+>.*?</[\w]+>|<[^<]*?>',re.I|re.S)
        dr=  dr.sub("",strSrcCode)
    return dr


FILE="F:\\bit-engd\\src-class\\data_language_all\\ASP\\ASP_2.txt"
#FILE="test.html"

with open(FILE,"r",encoding="utf-8") as f:
     str=f.read()


newresult=StripHTML(str)
print(newresult)
#result=pattern1.findall(newresult)


'''

cr = re.compile(r'<[^>]*>.*?</[^>]*>')
s = 'adffd<img>sdafdsf</img>'
print(cr.search(s))
'''