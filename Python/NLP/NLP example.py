from nltk.corpus import brown
brown.words()
brown.categories()
from nltk.book import *
review_words = brown.words(categories ='reviews')
review_words
len(review_words)

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
example_word = "Hello this is my first nltk code"
example_sent = "Hello this is my first nltk code"

print(word_tokenize(example_word))
print(sent_tokenize(example_sent))

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))



sent1 = 'an apple a day keeps diseases at bay'
word_token = word_tokenize(sent1)
word_token

# filtered_sent1 = [w for w in word_token if not w in stop_words]
filtered_sent1 = []
for w in word_token:
    if w not in stop_words: 
        filtered_sent1.append(w)

print(filtered_sent1)

from nltk.stem import PorterStemmer
ps = PorterStemmer()
new_text = 'Importance of caving as explained by cavers'
 
words = word_tokenize(new_text)
for w in words: 
    print(ps.stem(w))
     
    
txt = """
    Text mining is also referred to as text data mining, 
    roughly equivalent to text analytics, is the process of deriving high_quality information from text. 
    High quality information is typically derived through he devising of patterns and trends through means 
    such as statistical pattern learning. 

"""

from nltk import pos_tag
sent_token = sent_tokenize(txt)
for i in sent_token: 
    wordslist = word_tokenize(i)
    wordslist = [w for w in wordslist if not w in stop_words]
    tagged = pos_tag(wordslist)

print(tagged)


















