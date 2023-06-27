# Natural Language Processing 

# Importing the Datasets
dataset = read.delim("C:/Users/snapp/Data-Projects/Datasets/Restaurant_Reviews.tsv", quote = '', stringsAsFactors = FALSE)

# Cleaning the Text 
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset$Review))
# See the First Review 
review = as.character(corpus[[1]])

# Change all the word from uppercase to lowercase 
corpus = tm_map(corpus, content_transformer(tolower))

# Remove all the numbers 
corpus = tm_map(corpus, removeNumbers)

# Remove all the punctuation 
corpus = tm_map(corpus, removePunctuation)

# Remove Nonrelevant Words like: it, is, this, that, the, ... 
corpus = tm_map(corpus, removeWords, stopwords())

# Stemming: extract root of the word
corpus = tm_map(corpus, stemDocument)

# Remove all the extra spaces 
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model 
dtm = DocumentTermMatrix(corpus)

# To Reduce Sparsity we can filter most frequent words 
dtm = removeSparseTerms(dtm, 0.999)

# ML Model that is good for NLP: NB, DT, RF 
df = as.data.frame(as.matrix(dtm))
df$Liked = dataset$Liked

# Encoding the Target Feature aas Factor 
df$Liked = factor(df$Liked, levels = c(0,1))

# Split Dataset into train test split 
library(caTools)
# Set random state for seeing same data 
set.seed(123)

split = sample.split(df$Liked, SplitRatio = 0.8)
train_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Feature Scaling is not necessary 
# Fit Random Forest To the training set and Predict the test set results  
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = train_set[-692],
                          y =train_set$Liked,
                          ntree = 10)

# Predict Test Results 
y_pred = predict(classifier, newdata = test_set[-692])

# Make Confusion Matrix 
CM = table(test_set[,692], y_pred)
print(CM)



