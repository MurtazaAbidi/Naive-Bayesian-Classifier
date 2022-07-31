# Pre-processing 
# Training Our Model on 80% of the DataSet
# while the remaining 20% of DataSet will be used for Testing Purpose

import math
from bs4 import BeautifulSoup
import os
import pickle
import string
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

def remove_punctuation(str):
    exclude = set(string.punctuation)
    for ch in str:
        if ch in exclude:
            str = str.replace(ch," ")
    return str

docs = []
distinct_term = [] 
files_path = "course-cotrain-data\\fulltext\\course"


for i in range (2):
    if (i == 1):
        files_path = "course-cotrain-data\\inlinks\\course"
    counter = 0
    training = len(os.listdir(files_path)) * 0.8
    print ("Total number of Documents:",int(training))

    for filename in os.listdir(files_path):
        with open(os.path.join(files_path, filename), 'r') as f:
            text = f.read()
            Soup = BeautifulSoup(text, 'html.parser')
            str = Soup.get_text()
            str = remove_punctuation(str)
            str = str.lower()
            str = (str).split()
            str = [word for word in str if word not in stop_words]
            docs.append(str)
            distinct = []
            distinct = [word.lower() for word in str if word not in distinct_term]
            distinct_term.extend (distinct)
            counter += 1
            if (counter > training):
                break

document_frequency_for_each_term = [0 for i in distinct_term if(True)]

print ("Total Distinct-terms:",len(distinct_term))
term_frequency = []
idf = []
temp = [] 
for i in range (len(docs)):
    for j in range (len(distinct_term)):
        count = docs[i].count(distinct_term[j])
        if count >= 1:
            document_frequency_for_each_term[j] += 1
        temp.append(count)
    term_frequency.append(temp)
    temp=[]
for i in document_frequency_for_each_term:
    i_df = float(math.log(len(docs)/i,10))
    idf.append(i_df)

tf_idf = []
for i in range (len(term_frequency)):
    doc_tfidf = [] 
    for j in range (len(term_frequency[0])):
        temp = (term_frequency[i][j] * idf[j])
        doc_tfidf.append(temp)
    tf_idf.append(doc_tfidf)


threshold = 10
top_features = []
temp_string = ''
for i in range (len(tf_idf)):
    for j in range (len (tf_idf[0])):
        if (tf_idf[i][j] > threshold and distinct_term[j] not in top_features):
            top_features.append(distinct_term[j])
            temp_string += distinct_term[j]+' '
top_features.sort()

nouns = [word for (word, pos) in nltk.pos_tag(top_features) if(pos[:2] == 'NN')]


dic = {} 
scores = [] 
for i in nouns:
    key = i 
    count = 0
    for j in range(len(docs)):
        if (docs[j].count(key)>0):
            count+=1
    dic[key]= count
    scores.append(count)
print (dic)
scores.sort(reverse=True)
print (scores)

if len(scores)>=50:
    threshold = scores[50]

course = [] 
dictionary = {} 
for i in dic:
    if (dic[i] >= threshold ):
        dictionary[i]=dic[i]
        course.append(i)

print (dictionary)
print (len (dictionary))


# ----------- for non course ---------

print ("\nFor Non-Course")
n_docs = []
n_distinct_term = []
n_files_path = "course-cotrain-data\\fulltext\\non-course"


for i in range (2):
    if (i == 1):
        n_files_path = "course-cotrain-data\\inlinks\\non-course"
    counter = 0
    n_training = len(os.listdir(n_files_path)) * 0.8
    
    print ("Total number of Documents:",int(n_training))

    for filename in os.listdir(n_files_path):
        with open(os.path.join(n_files_path, filename), 'r') as f:
            n_text = f.read()
            Soup = BeautifulSoup(n_text, 'html.parser')
            n_str = Soup.get_text()
            n_str = remove_punctuation(n_str)
            n_str = n_str.lower()
            n_str = (n_str).split()
            n_str = [word for word in n_str if word not in stop_words]
            n_docs.append(n_str)
            n_distinct = []
            n_distinct = [word.lower() for word in n_str if word not in n_distinct_term]
            n_distinct_term.extend (n_distinct)
            counter += 1
            if (counter > n_training):
                break

n_document_frequency_for_each_term = [0 for i in n_distinct_term if(True)]
print ("Total Distinct-terms:",len(n_distinct_term))
n_term_frequency = []
n_idf = []
temp = [] 
for i in range (len(n_docs)):
    for j in range (len(n_distinct_term)):
        count = n_docs[i].count(n_distinct_term[j])
        if count >= 1:
            n_document_frequency_for_each_term[j] += 1
        temp.append(count)
    n_term_frequency.append(temp)
    temp=[]
for i in n_document_frequency_for_each_term:
    i_df = float(math.log(len(n_docs)/i,10))
    n_idf.append(i_df)

n_tf_idf = []
for i in range (len(n_term_frequency)):
    n_doc_tfidf = [] 
    for j in range (len(n_term_frequency[0])):
        temp = (n_term_frequency[i][j] * n_idf[j])
        n_doc_tfidf.append(temp)
    n_tf_idf.append(n_doc_tfidf)

threshold = 10
n_top_features = []
for i in range (len(n_tf_idf)):
    for j in range (len (n_tf_idf[0])):
        if (n_tf_idf[i][j] > threshold and n_distinct_term[j] not in n_top_features):
            n_top_features.append(n_distinct_term[j])
n_top_features.sort()


n_nouns = [word for (word, pos) in nltk.pos_tag(n_top_features) if(pos[:2] == 'NN')]

n_dic = {} 
n_scores = [] 
for i in n_nouns:
    key = i 
    count = 0
    for j in range(len(n_docs)):
        if (n_docs[j].count(key)>0):
            count+=1
    n_dic[key]= count
    n_scores.append(count)

n_scores.sort(reverse=True)

if len(n_scores)>=50:
    threshold = n_scores[50]

n_course = [] 
n_dictionary = {} 
for i in n_dic:
    if (n_dic[i] >= threshold ):
        n_dictionary[i]=n_dic[i]
        n_course.append(i)


totalFeatures = [] 
for i in dictionary:
    totalFeatures.append(i)
for i in n_dictionary:
    if i not in totalFeatures:
        totalFeatures.append(i)


Prob_Course = training/(training+n_training)
prob_NonCourse = n_training/(training+n_training)


c = {}
for i in dictionary:
    c[i] = Prob_Course * (dictionary[i]+1)/(training+len(totalFeatures))

for i in n_dictionary:
    if i not in course:
        c[i] = Prob_Course * 1/(training+len(totalFeatures))

print (c)
nc = {}
for i in n_dictionary:
    nc[i] = prob_NonCourse * (n_dictionary[i]+1)/(n_training+len(totalFeatures))
for i in dictionary:
    if i not in n_course:
        nc[i] = prob_NonCourse * 1/(n_training+len(totalFeatures))
print (nc)

courseFile = open("Course.pkl", "wb")
pickle.dump(c, courseFile)
courseFile. close()

nonCourseFile = open ("Non-Course.pkl", "wb")
pickle.dump(nc, nonCourseFile)

# Course.pkl : consist of the probablities of each term given that it is in course class (i.e. P(term|Course))
# Non-Course.pkl : consist of the probablities of each term given that it is in course class (i.e. P(term|non-Course))