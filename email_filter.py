import numpy
from email_process import read_bagofwords_dat
import sklearn 
import sys
import os

def bagofwords_tran(myfile,numofemails):
    bagofwords = read_bagofwords_dat(myfile,numofemails)
    return bagofwords

def label_tran(myfile,numofemails):
    label_array = numpy.genfromtxt(myfile,dtype='str')
    if(numofemails != numpy.size(label_array)):
        print "num of emails doesn't match!"
    # class_array = numpy.ones(numofemails)
    # for i in range(numofemails):
    #    if cmp(label_array[i],"NotSpam")==0 :
    #        class_array[i] = 0
    #    else:
    #        class_array[i] = 1
    return label_array
