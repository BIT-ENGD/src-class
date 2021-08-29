
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


lines=["abc\n"]
lines=list(map(lambda x:x.replace("\n",""),lines))
print(lines )

test = [32, 39, 45, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 46, 47, 58, 59, 63, 64, 91, 92, 10, 9, 93, 94, 95, 96, 123, 124, 125, 126, 43, 60, 61, 62, 215, 247, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 97, 65, 98, 66, 99, 67, 100, 68, 101, 69, 102, 70, 103, 71,104, 72, 105, 73, 106, 74, 107, 75, 108, 76, 109, 77, 110, 78, 111, 79, 112, 80, 113, 81, 114, 82, 115, 83, 116, 84, 117, 85, 118, 86,119, 87, 120, 88, 121, 89, 122, 90]

for item in test:
    print(chr(item))