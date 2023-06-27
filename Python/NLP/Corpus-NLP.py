#import nltk library 
import nltk 
#read file 
with open("C:/Users\snapp/Downloads/archive/brown/brown/ca10",'r') as myfile:
    data = myfile.read().replace('\n','')

data2 = data.replace('/','')
for i, line in enumerate(data2.split('\n')):
    if i>10: 
        break 
    print(str(i) + ':\t' + line) 
    
#perform tokenization
from nltk import sent_tokenize, word_tokenize    
sent_tokenize(data2)
for sent in sent_tokenize(data2): 
    print(word_tokenize(sent))

#Remove stopwords    
from nltk.corpus import stopwords
stopwords_en = stopwords.words('english')
single_tokenized_lowered = list(map(str.lower, word_tokenize(data2)))
print(single_tokenized_lowered)


stopwords = set(stopwords.words('english'))

print([word for word in single_tokenized_lowered if word not in stopwords_en ])

from string import punctuation 
print('From string.punctuation:',type(punctuation),punctuation)

stopwords_en_withpunct = stopwords.union(set(punctuation))
print(stopwords_en_withpunct)


print([word for word in single_tokenized_lowered if word not in stopwords_en_withpunct])

#perform stemming and lemmatization 
from nltk.stem import PorterStemmer
porter = PorterStemmer()
for word in single_tokenized_lowered:
    print(porter.stem(word))

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

for word in single_tokenized_lowered:
    print(wnl.lemmatize(word))

##stop_words = set(stopwords.words('english'))
tokenized = sent_tokenize(data2)
for i in tokenized:
    wordslist = nltk.word_tokenize(i)
    wordlist = [w for w in wordslist if not w in stopwords]
    tagged = nltk.pos_tag(wordslist)
    
    print(tagged)

#Perform Named Entity Recognition 
sentences = nltk.sent_tokenize(data2)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
chunked_sentences = [ nltk.ne_chunk_sents(tagged_sentences, binary = True )]

def extract_entity_names(t): 
    entity_names = []
    if hasattr(t,'label') and t.label: 
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
           for child in t: 
               entity_names.extend(extract_entity_names(child))
    return entity_names 

entity_names = []
for tree in chunked_sentences:
    entity_names.extend(extract_entity_names(tree))

print(set(entity_names))

































