# FakeNewsClassifier

![1*LBn_F2ALsKUAKK6vbQbJuw](https://user-images.githubusercontent.com/100385953/179432957-feebd5a3-4c41-4e33-b8dc-120abb296c11.png)


News medium has become a channel to pass on the information of what’s happening on world to the people living. Often people perceive whatever conveyed in the news to be true. There were circumstances where even the news channels acknowledged that their news is not true as they wrote. But some news have a significant impact not only to the people or government but also the economy. One news can shift the curves up and down depending on the emotions of people and political situation. It is important to identify the fake news from the real true news. The problem has been taken over and resolved with the help of Natural Language Processing tools which help us identify fake or true news based on the historical data. The news are now in safe hands !
Problem statement


The authenticity of Information has become a longstanding issue affecting businesses and society, both for printed and digital media. On social networks, the reach and effects of information spread occur at such a fast pace and so amplified that distorted, inaccurate or false information acquires a tremendous potential to cause real world impacts, within minutes, for millions of users. Recently, several public concerns about this problem and some approaches to mitigate the problem were expressed. . The sensationalism of not-so-accurate eye catching and intriguing headlines aimed at retaining the attention of audiences to sell information has persisted all throughout the history of all kinds of information broadcast. On social networking websites, the reach and effects of information spread are however significantly amplified and occur at such a fast pace, that distorted, inaccurate or false information acquires a tremendous potential to cause real impacts, within minutes, for millions of user


Objective
Our sole objective is to classify the news from the dataset to fake or true news.
Extensive EDA of news
Selecting and building a powerful model for classification


![1*RGVPc-MT0q_DCHCavFRHvA](https://user-images.githubusercontent.com/100385953/179433047-e9dfc115-9711-43f9-b44d-ec6619b7e95a.jpeg)

Photo: Medium



Procedure: Medium




Dataset
Kaggle Data


train.csv: A full training dataset with the following attributes:
id: unique id for a news article
title: the title of a news article
author: author of the news article
text: the text of the article; could be incomplete
label: a label that marks the article as potentially unreliable. 
Where 1: unreliable and 0: reliable.
Reading the data
import pandas as pd
train = pd.read_csv('train.csv')
train.head()




Here’s how the training data looks like
We can see that the features ‘title’, ‘author’ and ‘text’ are important and all are in text form. So, we can combine these features to make one final feature which we will use to train the model. Let’s call the feature ‘total’.

# Firstly, fill all the null spaces with a space
train = train.fillna(' ')
train['total'] = train['title'] + ' ' + train['author'] + ' ' +
                 train['text']

After adding the column ‘total’, the data looks like this





Pre-processing/ Cleaning the Data
For preprocessing the data, we will need some libraries.
import ntlk
from ntlk.corpus import stopwords
from ntlk.stem import WordNetLemmatizer





The uses of all these libraries are explained below.
    # 1. Stopwords: Stop words are those common words that appear in a text many times and do not contribute to machine’s understanding of the text.
We don’t want these words to appear in our data. So, we remove these words.


All these stopwords are stored in the ntlk library in different languages.

stop_words = stopwords.words('english')

   # 2. Tokenization: Word tokenization is the process of splitting a large sample of text into words.
For example:
word_data = "It originated from the idea that there are readers who prefer learning new skills from the comforts of their drawing rooms"
nltk_tokens = nltk.word_tokenize(word_data)
print(ntlk_tokens)

It will convert the string word_data into this:
[‘It’, ‘originated’, ‘from’, ‘the’, ‘idea’, ‘that’, ‘there’, ‘are’, ‘readers’, ‘who’, ‘prefer’, ‘learning’, ‘new’, ‘skills’, ‘from’, ‘the’, ‘comforts’, ‘of’, ‘their’, ‘drawing’, ‘rooms’]


   # 3. Lemmatization: Lemmatization is the process of grouping together the different inflected forms of same root word so they can be analysed as a single item.
Examples of lemmatization:
swimming → swim
rocks → rock
better → good
lemmatizer = WordNetLemmatizer()
The code below is for lemmatization for our test data which excludes stopwords at the same time.
for index, row in train.iterrows():
    filter_sentence = ''
    sentence = row['total']
    # Cleaning the sentence with regex
    sentence = re.sub(r'[^\w\s]', '', sentence)
   
   
   # 4. Tokenization
    words = nltk.word_tokenize(sentence)
    # Stopwords removal
    words = [w for w in words if not w in stop_words]
    # Lemmatization
    for words in words:
        filter_sentence = filter_sentence  + ' ' +
                         str(lemmatizer.lemmatize(words)).lower()
    train.loc[index, 'total'] = filter_sentence
train = train[['total', 'label']]

This is how the data looks after pre-processing
X_train = train['total']
Y_train = train['label']
Finally, we have pre-processed the data but it is still in text form and we can’t provide this as an input to our machine learning model. We need numbers for that. How can we solve this problem? The answer is Vectorizers.
Vectorizer
For converting this text data into numerical data, we will use two vectorizers.



   # 5. Count Vectorizer
In order to use textual data for predictive modelling, the text must be parsed to remove certain words — this process is called tokenization. These words need to then be encoded as integers, or floating-point values, for use as inputs in machine learning algorithms. This process is called feature extraction (or vectorization).

TF-IDF Vectorizer


TF-IDF stands for Term Frequency — Inverse Document Frequency. It is one of the most important techniques used for information retrieval to represent how important a specific word or phrase is to a given document.


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)


The code written above will provide with you a matrix representing your text. It will be a sparse matrix with a large number of elements in Compressed Sparse Row format.
