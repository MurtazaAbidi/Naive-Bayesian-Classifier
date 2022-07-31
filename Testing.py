import os
from bs4 import BeautifulSoup
import string
import pickle
import eel
from nltk.corpus import stopwords


def remove_punctuation(str):
    exclude = set(string.punctuation)
    for ch in str:
        if ch in exclude:
            str = str.replace(ch," ")
    return str

@eel.expose
def testing(): 
    with open('course.pkl', 'rb') as f:
        course = pickle.load(f)

    with open('non-course.pkl', 'rb') as f:
        non_course = pickle.load(f)


    features_list = []
    for i in course:
        features_list.append(i)

    file_path = "course-cotrain-data\\fulltext\\course"
    testing = len(os.listdir(file_path)) * 0.2
    training = len(os.listdir(file_path))-testing
    stop_words = set(stopwords.words('english'))
    courseDocs = []
    for i in range (2):
        counter = 0
        if (i == 1):
            file_path = "course-cotrain-data\\inlinks\\course"
        for filename in os.listdir(file_path):
                if counter < len(os.listdir(file_path)) - (testing): 
                    counter+=1
                    continue
                with open(os.path.join(file_path, filename), 'r') as f:
                    text = f.read()
                    Soup = BeautifulSoup(text, 'html.parser')
                    str = Soup.get_text()
                    str = remove_punctuation(str)
                    str = str.lower()
                    str = (str).split()
                    str = [word for word in str if word not in stop_words]
                    courseDocs.append(str)



    file_path = "course-cotrain-data\\fulltext\\non-course"
    n_testing = len(os.listdir(file_path)) * 0.2
    n_training = len(os.listdir(file_path))-n_testing

    n_courseDocs = []
    for i in range (2):
        counter = 0
        if (i == 1):
            file_path = "course-cotrain-data\\inlinks\\non-course"
        for filename in os.listdir(file_path):
                if counter < len(os.listdir(file_path)) - (n_testing): 
                    counter+=1
                    continue
                with open(os.path.join(file_path, filename), 'r') as f:
                    text = f.read()
                    Soup = BeautifulSoup(text, 'html.parser')
                    n_str = Soup.get_text()
                    n_str = remove_punctuation(n_str)
                    n_str = n_str.lower()
                    n_str = (n_str).split()
                    n_str = [word for word in n_str if word not in stop_words]
                    n_courseDocs.append(n_str)

    testing_courseFiles = [] 
    for i in courseDocs: 
        if i in courseDocs:
            testing_courseFiles.append(i)

    Prob_Course = training/(training+n_training)
    prob_NonCourse = n_training/(training+n_training)

    print ()
    print ()
    print ("Testing for course Documents")
    docs_dicts = []
    c_documents={}
    for i in range (len(courseDocs)):
        c_documents = {} 
        for j in range (len (courseDocs[i])):
            if courseDocs[i][j] in features_list:
                c_documents[courseDocs[i][j]] = courseDocs[i].count(courseDocs[i][j])
        docs_dicts.append(c_documents)

    prob_in_course = {}
    for i in range (len (docs_dicts)):
        dic = docs_dicts[i]
        ans = Prob_Course
        for j in docs_dicts[i]:
            ans *= (course[j]**docs_dicts[i][j])
        prob_in_course[i+1] = ans

    prob_in_non_course = {}
    for i in range (len (docs_dicts)):
        dic = docs_dicts[i]
        ans = prob_NonCourse
        for j in docs_dicts[i]:
            ans *= (non_course[j]**docs_dicts[i][j])
        prob_in_non_course[i+1] = ans
    cc = 0
    nn = 0 
    for i in prob_in_course:
        if (prob_in_course[i]>=prob_in_non_course[i]):
            cc+=1
            print ("doc",i," ----> course", prob_in_course[i],prob_in_non_course[i])
        else:
            nn+=1
            print ("doc",i," ----> non-course", prob_in_course[i],prob_in_non_course[i])

    print ()
    print ()
    print ("Testing for Non-course Documents")
    n_docs_dicts = []
    n_c_documents={}
    for i in range (len(n_courseDocs)):

        n_c_documents = {} 
        for j in range (len (n_courseDocs[i])):
            if n_courseDocs[i][j] in features_list:
                n_c_documents[n_courseDocs[i][j]] = n_courseDocs[i].count(n_courseDocs[i][j])
        n_docs_dicts.append(n_c_documents)

    n_prob_in_course = {}
    for i in range (len (n_docs_dicts)):
        n_dic = n_docs_dicts[i]
        ans = Prob_Course
        for j in n_docs_dicts[i]:
            ans *= (course[j]**n_docs_dicts[i][j])
        n_prob_in_course[i+1] = ans

    n_prob_in_non_course = {}
    for i in range (len (n_docs_dicts)):
        dic = n_docs_dicts[i]
        ans = prob_NonCourse
        for j in n_docs_dicts[i]:
            ans *= (non_course[j]**n_docs_dicts[i][j])
        n_prob_in_non_course[i+1] = ans
    cc_forNonCourse = 0
    nn_forNonCourse = 0 
    for i in n_prob_in_course:
        if (n_prob_in_course[i]>=n_prob_in_non_course[i]):
            cc_forNonCourse+=1
            print ("doc",i," ----> course", n_prob_in_course[i],n_prob_in_non_course[i])
        else:
            nn_forNonCourse+=1
            print ("doc",i," ----> non-course", n_prob_in_course[i],n_prob_in_non_course[i])
            
    TP = cc
    TN = nn_forNonCourse
    FP = nn
    FN = cc_forNonCourse
    print ("Model Predicted Course docs that are also in Course Class:", cc)
    print ("Model Predicted Non-Course docs that are acctually in Course Class:", nn)
    print ("Model Predicted Course docs that are acctually in Non-Course Class:", cc_forNonCourse)
    print ("Model Predicted Non-Course docs that are also in non-Course Class:", nn_forNonCourse)
    print ()

    print ("    TP=",TP,"      |       FP=",FP,"      ")
    print ("    FN=",FN,"      |       TN=",TN,"      \n")
    print ()
    accuracy = (TP + TN)/(TP + TN + FP +FN)
    accuracy=round(accuracy*100,2)
    print ('Accuracy =' , accuracy, '%\n')

    Recall_course = TP / (TP+FN)
    rec1=round(Recall_course*100,2)
    print ('Recall(course) = ',rec1 , '%' )

    Precision_course = TP / (TP+FP)
    prec1=round(Precision_course*100,2)
    print ('Precision(course) = ',prec1 , '%')

    F1_measure_course = (2*Precision_course*Recall_course)/(Precision_course+Recall_course)
    F1_measure_course = round(F1_measure_course*100,2)
    print ('F1_measure(course) = ',F1_measure_course , '%\n')

    Recall_nonCourse = TN / (FP+TN)
    rec2= round(Recall_nonCourse*100,2)
    print ('Recall(non-course) = ',rec2 , '%' )

    Precision_nonCourse = TN / (FN+TN)
    prec2=round(Precision_nonCourse*100,2)
    print ('Precision(non-course) = ',prec2 , '%')

    F1_measure_nonCourse = (2*Precision_nonCourse*Recall_nonCourse)/(Precision_nonCourse+Recall_nonCourse)
    F1_measure_nonCourse = round(F1_measure_nonCourse*100,2)
    print ('F1_measure(non-course) = ',F1_measure_nonCourse , '%')

    
    eel.showPerformance(accuracy, rec1, prec1, F1_measure_course, rec2, prec2, F1_measure_nonCourse)


eel.init("frontend")
eel.start("index.html")