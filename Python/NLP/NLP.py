import os 
import nltk 
import nltk.corpus

#Diffrent files that are in the nltk.corpus
print(os.listdir(nltk.data.find("corpora")))
from nltk.corpus import brown 
brown.words()

nltk.corpus.gutenberg.fileids()

# take the hamlet text
hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
hamlet
# Look at 50o words of the hamlet 
for word in hamlet[:500]:
    print(word, sep = ' ', end = ' ')
    
AI = """According to the father of Artificial Intelligence, John McCarthy, it is “The science and engineering of making intelligent machines, especially intelligent computer programs”.

Artificial Intelligence is a way of making a computer, a computer-controlled robot, or a software think intelligently, in the similar manner the intelligent humans think.

AI is accomplished by studying how human brain thinks, and how humans learn, decide, and work while trying to solve a problem, and then using the outcomes of this study as a basis of developing intelligent software and systems."""

type(AI)

# Tokenize Text
from nltk.tokenize import word_tokenize 
AI_tokens = word_tokenize(AI)
print(AI_tokens)
len(AI_tokens)

# Calculate the frequency of the words in the text
from nltk.probability import FreqDist
fdist = FreqDist()
for word in AI_tokens: 
    fdist[word.lower()]+=1
fdist

fdist['artificial']

# instead of 102 we have 60 real part in text 
len(fdist)

fdist_top10 = fdist.most_common(10)
fdist_top10

#Number of paragraph which are seprated by a new line 
from nltk.tokenize import blankline_tokenize
AI_blank = blankline_tokenize(AI)
len(AI_blank)

AI_blank[2]

# Check bigrams - trigrams - ngrams 
from nltk.util import bigrams,trigrams, ngrams

string = "The best and the most beautiful things in the world can not be seen or even touched, they muust be felt by heart"
quotes_token = nltk.word_tokenize(string)
quotes_token

quotes_bigrams = list(nltk.bigrams(quotes_token))
quotes_bigrams

quotes_trigrams = list(nltk.trigrams(quotes_token))
quotes_trigrams

quotes_ngrams = list(nltk.ngrams(quotes_token,5))
quotes_ngrams

# ِDifferent Stemming (words into their bases)
# Output of stemming might be a proper word 
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer

pst = PorterStemmer()
lst = LancasterStemmer()
sbst = SnowballStemmer('english')

words_to_stem=['give','given','giving','gave']

for words in words_to_stem:
    print(words + ":" + pst.stem(words))

for words in words_to_stem:
    print(words + ":" + lst.stem(words))
    
for words in words_to_stem:
    print(words + ":" + sbst.stem(words))

# Lemmatization 
# Output of lemmatization is a proper word 
from nltk.stem import wordnet, WordNetLemmatizer
word_lem = WordNetLemmatizer()
word_lem.lemmatize('corpora')

for words in words_to_stem:
    print(words + ":" + word_lem.lemmatize(words))

#Stop Words
from nltk.corpus import stopwords
stopwords.words('english')
len(stopwords.words('english'))

# Remove Punctuation from text 
import re  
punctuation = re.compile(r'[“.:,;”?!]')
post_punctatuion = []
for words in AI_tokens: 
    word = punctuation.sub("",words)
    if len(word)>0:
        post_punctatuion.append(word)

post_punctatuion

#POS : part of speech 
sent = "Timoty is a natural when it comes to drawing"
sent_token = word_tokenize(sent)
for token in sent_token:
    print(nltk.pos_tag([token]))

# NER: Named Entity Reconition - Movie, Person, Location, Organization
from nltk import ne_chunk
sent2 = "The US president stay in the White House"
sent2_token = word_tokenize(sent2)
sent2_tags = nltk.pos_tag(sent2_token)
sent2_ner = ne_chunk(sent2_tags)

print(sent2_ner)





































