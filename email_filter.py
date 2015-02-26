import numpy
from email_process import read_bagofwords_dat
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sys
import os
import time

def bagofwords_tran(myfile,numofemails):
    bagofwords = read_bagofwords_dat(myfile,numofemails)
    return bagofwords

def label_tran(myfile,numofemails):
    label_array = numpy.genfromtxt(myfile,dtype='str')
    if(numofemails != numpy.size(label_array)):
        print "num of emails doesn't match!"
    class_array = numpy.ones(numofemails)
    for i in range(numofemails):
        if cmp(label_array[i],"NotSpam")==0 :
            class_array[i] = 0
        else:
            class_array[i] = 1
    return class_array

def vocab_tran(myfile):
    vocab_array = numpy.genfromtxt(myfile,dtype='str')
    return vocab_array

def feature_select(train_data,train_target,test_data,feature_names):
    svc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_data,train_target)
    train_data_new = svc.transform(train_data)
    test_data_new = svc.transform(test_data)
    select_feature_names  = numpy.asarray(feature_names[numpy.flatnonzero(svc.coef_)])
    return train_data_new,test_data_new,select_feature_names

def multinomial_nb(train_data,train_target,test_data):
    mnb = MultinomialNB()
    y_pred = mnb.fit(train_data,train_target).predict(test_data)
    return y_pred

def kdtree_knn(train_data,train_target,test_data):
    neigh = KNeighborsClassifier(algorithm='kd_tree')
    neigh.fit(train_data,train_target)
    y_pred = neigh.predict(test_data)
    return y_pred

def l2_lr(train_data,train_target,test_data):
    lr = LogisticRegression(penalty='l2')
    y_pred = lr.fit(train_data,train_target).predict(test_data)
    return y_pred

def l2_perceptron(train_data,train_target,test_data):
    p2 = Perceptron(penalty='l2')
    y_pred = p2.fit(train_data,train_target).predict(test_data)
    return y_pred

def l2_hingeloss(train_data,train_target,test_data):
    hl2 = SGDClassifier(loss='hinge',penalty='l2')
    y_pred = hl2.fit(train_data,train_target).predict(test_data)
    return y_pred

def linear_svm(train_data,train_target,test_data):
    svml = LinearSVC(loss='l1',penalty='l2')
    y_pred = svml.fit(train_data,train_target).predict(test_data)
    return y_pred

def adaboost(train_data,train_target,test_data):
    ada  = AdaBoostClassifier(n_estimators=100)
    y_pred = ada.fit(train_data,train_target).predict(test_data)
    return y_pred

def decision_tree(train_data,train_target,test_data):
    dt  = DecisionTreeClassifier(criterion="gini")
    y_pred = dt.fit(train_data,train_target).predict(test_data)
    return y_pred


def random_forest(train_data,train_target,test_data):
    rf  = RandomForestClassifier(n_estimators=100,criterion="gini")
    y_pred = rf.fit(train_data,train_target).predict(test_data)
    return y_pred

#def multinomial_nb_select(train_data,train_target,test_data,feature_names):
#    mnb = MultinomialNB()
#    train_data_select,test_data_select,select_feature_names  = feature_select(train_data,train_target,test_data,feature_names)
#    y_pred = mnb.fit(train_data_select,train_target).predict(test_data_select)
#    return y_pred,select_feature_names

def evaluate(test_class,pred_class):
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    for i in range(numpy.size(test_class)):
        if(pred_class[i]==1):
            if(test_class[i]==1):
                TP += 1;
            else:
                FP += 1;
        else:
            if(test_class[i]==0):
                TN += 1;
            else:
                FN += 1;
    if(TP+FP+TN+FN != numpy.size(test_class)):
        print "sum of ROC is not correct!"
    precision = TP/(float)(TP+FP)
    recall = TP/(float)(TP+FN)
    F1 = 2*((precision*recall)/(precision+recall))
    return precision, recall, F1

def main():
    train_data = bagofwords_tran("./trec07p_data/Train/train_emails_bag_of_words_200.dat",45000)
    train_class = label_tran("./trec07p_data/Train/train_emails_classes_200.txt",45000)
    test_data = bagofwords_tran("./trec07p_data/Test/test_emails_bag_of_words_0.dat",5000)
    test_class = label_tran("./trec07p_data/Test/test_emails_classes_0.txt",5000)
    vocab = vocab_tran("./trec07p_data/Train/train_emails_vocab_200.txt")
    start_time = time.time()
    pred_class = kdtree_knn(train_data,train_class,test_data)
    #print "vocab size:",vocab.shape
    #print "selected_vocab size:",selected_vocab.shape
    #print "train_class size:",train_class.size
    #print "pred_class size:",pred_class.size
    time_used = time.time()-start_time
    precision,recall,F1 = evaluate(test_class,pred_class)
    print "precision:", precision
    print "time:", time_used
    #print "selected_vocab:",selected_vocab
    #print "vocab size:",vocab.shape,selected_vocab.shape
    return precision

if __name__ == "__main__":
    main()
