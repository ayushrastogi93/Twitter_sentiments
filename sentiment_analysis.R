install.packages("plyr")
install.packages("dplyr")
install.packages("tm")
install.packages('caTools')
library(dplyr)
library(plyr)
library(tm)
library(ggplot2)
library(caTools)
set.seed(123)
z1= read.csv("D:/ML_datasets/Twitter_sentiments/train_tweets.csv")
z3= read.csv("D:/ML_datasets/Twitter_sentiments/train_tweets3.csv")
z2 = read.csv("D:/ML_datasets/Twitter_sentiments/train_tweets2.csv")
z_all = join_all(list(z1,z2,z3),type= "full", by="id")
library(tm)
library(SnowballC)# to get list of stop words
# check for dimension of data 
#dim(z_all) and glimpse(z_all) and count(z_all$label) 
#head of data 
#head(z_all)
text_clean <- function(crudedata){
  coupus = na.omit(crudedata)# remove NA data
  corpus <- iconv(crudedata$tweet , to ="utf-8")
  corpus = Corpus(VectorSource(corpus))#review all our data
  corpus = tm_map(corpus, tolower)# all reviews in lower
  corpus = tm_map(corpus, removeNumbers)#remove nos. 
  corpus = tm_map(corpus, removePunctuation)# remove punctions 
  corpus = tm_map(corpus, removeWords, stopwords())# remove un-necessary words (such/as/the/is/are)
  removeURL<- function(x)gsub('http[[:alnum:]]*',"",x)
  corpus = tm_map(corpus,content_transformer(removeURL))
  corpus = tm_map(corpus, stemDocument)
  corpus = tm_map(corpus, stripWhitespace)
 
   # Creating the Bag of Words model
  dtm = DocumentTermMatrix(corpus)
  dtm = removeSparseTerms(dtm, 0.999)#dtm is matrix
  refined_tweets = as.data.frame(as.matrix(dtm))#converting from matrix into dataframe .
  refined_tweets$label = z_all$label
  
  # Encoding the target feature as factor
  refined_tweets$label = factor(refined_tweets$label, levels = c(0, 1))
  return(refined_tweets)
  
}
#crudedata = na.omit(z_all)
refined_tweets = text_clean(z_all)
# Splitting the dataset into the Training set and Test set
split = sample.split(refined_tweets$tweet, SplitRatio = 0.8)

training_set = subset(refined_tweets, split == TRUE)
test_set = subset(refined_tweets, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier_rf = randomForest(x = training_set[-1163],
                          y = training_set$label,
                          ntree = 100)

# Predicting the Test set results
y_pred = predict(classifier_rf, newdata = test_set[-1163])

# Making the Confusion Matrix
cm = table(test_set[,3], y_pred)

