import pickle

VOCAB_DICT_FILE="vocab_dict.dat"



with open(VOCAB_DICT_FILE,"rb") as f:
    vocab=pickle.load(f)


total =0

new_vocab={}
index=1
for item in vocab:
    if vocab[item] >0:
        new_vocab[item]=index
        index+=1
        total +=1
print(total)

#print(max(vocab.values()))
print(new_vocab["java"])

new_vocab[""]=0

with open("vocab_real.dat","wb") as f:
     pickle.dump(new_vocab,f)