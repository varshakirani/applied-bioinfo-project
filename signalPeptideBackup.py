# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 23:47:36 2015

@author: varsha
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 19:59:24 2015

@author: varsha
"""

import hmm;
import inputRead as ir;
import copy;
import numpy as np;
from random import shuffle;
import math;
import sqlite3;
#obj = hmm.HMM();

A = [[0.4, 0.2, 0.2, 0.2],[0.2, 0.4, 0.2, 0.2],[ 0.2, 0.2, 0.4, 0.2],[ 0.2, 0.2, 0.2, 0.4]];
B = [[0.4, 0.2, 0.2, 0.2],[ 0.2, 0.4, 0.2, 0.2],[ 0.2, 0.2, 0.4, 0.2],[ 0.2, 0.2, 0.2, 0.4] ];
pi = [0.241896, 0.266086, 0.249153, 0.242864 ];

#A = np.array(A);
#B = np.array(B);
#pi = np.array(pi);
#
#obs_seq = [0,1,2,3,3,0,0,1,1,1,2,2,2,3,0,0,0,1,1,1,2,3,3,0,0,0,1,1,1,2,3,3,0,1,2,3,0,1,1,1,2,3,3,0,1,2,2,3,0,0,0,1,1,2,2,3,0,1,1,2,3,0,1,2,2,2,2,3,0,0,1,2,3,0,1,1,2,3,3,3,0,0,1,1,1,1,2,2,3,3,3,0,1,2,3,3,3,3,0,1,1,2,2,3,0,0,0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,3,3,3,3,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,0,1,2,3,0,1,1,1,2,3,0,1,1,2,2,2,2,2,3,0,1,1,1,2,2,2,2,3,0,0,0,0,0,1,1,1,1,2,2,3,3,0,1,2,3,3,0,0,0,0,0,0,1,1,2,2,3,0,0,1,1,1,1,1,1,2,3,3,0,0,1,1,1,2,3,0,0,1,2,3,0,1,1,2,3,3,0,0,0,1,2,3,3,3,0,1,1,1,1,2,3,3,3,3,3,3,0,1,2,2,2,2,2,2,3,0,1,1,1,2,2,3,3,3,3,0,1,2,3,0,0,0,1,1,2,2,3,0,0,0,0,0,0,0,1,2,2,2,3,3,3,3,0,0,1,2,2,2,3,3,3,0,0,1,2,2,3,0,0,0,0,1,1,1,2,3,3,3,3,3,3,3,3,0,1,2,3,0,0,1,2,3,3,3,0,0,0,0,0,1,1,1,1,2,3,0,0,0,1,2,2,3,3,0,0,0,1,1,1,1,1,2,3,3,3,3,0,1,1,1,2,2,3,0,1,2,3,3,3,3,0,0,0,0,1,2,3,3,0,1,2,2,3,3,0,0,1,1,2,3,3,0,1,2,2,3,3,3,0,0,1,1,2,3,3,3,3,0,0,1,1,2,3,3,0,1,2,3,0,1,1,2,2,3,0,1,2,3,3,0,1,1,1,2,2,2,3,3,0,0,1,1,1,1,1,2,3,3,3,0,1,1,2,2,2,2,3,3,0,0,1,2,3,0,1,1,2,2,2,2,3,0,0,1,2,2,3,0,0,0,0,0,1,1,1,2,3,0,0,1,2,3,3,0,0,0,1,2,2,2,3,3,0,0,0,1,2,2,2,2,2,3,0,1,1,2,3,0,0,1,1,1,2,2,3,0,0,0,0,1,1,1,2,2,3,0,1,1,1,2,2,2,3,3,0,0,1,2,2,3,3,3,0,1,1,2,3,0,0,0,0,0,1,2,2,2,3,3,3,0,0,0,1,2,3,0,1,1,2,3,3,3,0,1,2,2,2,3,0,0,1,1,1,1,2,3,3,0,0,0,0,1,2,3,3,3,0,0,0,1,1,2,3,0,1,1,1,1,2,2,2,2,2,2,3,0,0,0,0,1,2,2,2,2,3,0,1,2,2,3,0,1,2,3,0,1,2,3,0,0,0,1,1,2,2,3,3,0,1,1,1,1,2,2,3,3,0,1,1,1,2,2,2,3,3,3,0,1,1,2,3,3,0,1,2,3,0,0,0,0,1,2,3,0,0,0,0,0,0,1,2,2,3,3,0,0,1,2,3,0,1,2,2,3,0,0,0,1,1,2,2,2,2,2,3,3,3,3,3,0,1,2,2,3,3,3,3,3,0,0,1,1,2,2,3,0,0,1,2,2,3,3,3,0,0,0,1,2,2,2,2,3,3,0,1,2,3,0,0,1,1,1,2,2,3,0,0,1,1,2,2,2,3,3,0,0,1,1,1,1,1,2,3,3,3,0,1,2,2,2,2,3,3,3,3,3,3,0,0,0,0,0,0,1,2,3,0,0,1,1,1,2,3,0,0,1,1,2,2,2,2,3,3,3,0,1,1,2,2,2,3,3,0,0,0,0,0,0,1,2,2,3,3,0,0,0,0,0,0,1,2,3,3,3,0,1,1,1,2,2,2,2,2,3,3,3,0,1,2,2,2,3,3,3,3,0,0,0,0,1,2,3,3,3,3,3,3,0,0,1,1,1,1,2,3,0,1,2,3,0,1,1,2,3,3,3,0,0,0,0,1,1,2,3,3,3,3,0,0,1,1,1,2,2,2,2,2,2,3,3,0,0,0,1,2,3,0,0,1,1,2,2,3,3,3,3,3,0,0,1,2,2,2,2,3,0,0,1,1,1,1,1,2,3,3,0,0,1,1,1,2,3,3,3,0,0];
#obs_seq = np.array(obs_seq);


#obj.setMatrices(A,B,pi);
#(a,b,p) = obj.getMatrices();
#print "before convergence:";
#print "A:";
#print a;
#print "B:";
#print b;
#print "pi:";
#print p
#obj.converge(obs_seq,25);

#(a,b,p) = obj.getMatrices();
#print "after convergence:";
#print "A:";
#print a;
#print "B:";
#print b;
#print "pi:";
#print p

(posSamples,negSamples)= ir.getSamples();


#sample = posSamples[0];


#posList.append(obj);
#hmmObj = hmm.HMM();
#(a,b,p) = hmmObj.getMatrices();
#print "before convergence:";
#print "A:";
#print a;
#print "B:";
#print b;
#print "pi:";
#print p
#hmmObj.alphaPass(sample);
#hmmObj.betaPass(sample);
#hmmObj.gammaDigamma(sample);
#hmmObj.reestimate(sample);
#hmmObj.converge(sample,20);
#print "after convergence:";
#print "A:";
#print a;
#print "B:";
#print b;
#print "pi:";
#print p
negList = [];    
posList = [];  
def trainData(trainPos,trainNeg):
      
    train_len = 3;
    index = 0;
    for posSample in trainPos:
        hmmObj = copy.deepcopy(hmm.HMM());
        hmmObj.converge(posSample,500);
        posList.append(hmmObj);
        index += 1;
        print index;
    for negSample in trainNeg:
        hmmObj = copy.deepcopy(hmm.HMM());
        hmmObj.converge(negSample,500);
        negList.append(hmmObj);
        index += 1;
        print index;
       
    return(posList,negList);

#test_len = posLen - train_len;


    
#print "testing:"


avgPosValues = [];
avgNegValues = [];
def testData(testPos,testNeg):
    misClassified = 0;
    classified = 0;    
    maxProb = 0;
    whichClassPos = 0;  #0 for positive and 1 for negative
    whichClassNeg = 0;  #0 for positive and 1 for negative
    avgPos = 0;
    avgNeg = 0;
    valPos = 0;
    valNeg = 0;
    avgClass = 0; #0 for positive and 1 for negative
    avgClassified = 0;
    avgMisClassified = 0;
    for test in testPos:
        for obj in posList:
            prob = obj.forwardAlgorithm(test);
            valPos += prob;            
            if prob > maxProb:
                maxProb = prob;
                whichClassPos = 0;
        for obj in negList:
            prob = obj.forwardAlgorithm(test);
            valNeg += prob;
            if prob > maxProb:
                maxProb = prob;
                whichClassPos = 1;
                
        print "classified as:"
        if whichClassPos == 1:
            print "non-signal";
            misClassified += 1;
        else:
            print "signal";
            classified += 1;
            
        avgPos = valPos/len(posList);
        avgNeg = valNeg/len(negList);
        avgPosValues.append(avgPos);
        avgNegValues.append(avgNeg);
        print str(avgPos) + str(avgNeg);
        if avgPos > avgNeg:
            avgClass = 0;
            avgClassified += 1;
        else:
            avgClass = 1;
            avgMisClassified += 1;
    
    valPos = 0;
    valNeg = 0;        
    for test in testNeg:
        for obj in posList:
            prob = obj.forwardAlgorithm(test);
            valPos += prob;
            if prob > maxProb:
                maxProb = prob;
                whichClassNeg = 0;
        for obj in negList:
            prob = obj.forwardAlgorithm(test);
            valNeg += prob;            
            if prob > maxProb:
                maxProb = prob;
                whichClassNeg = 1;
                
        print "classified as:";
        if whichClassNeg == 1:
            print "non-signal";
            classified += 1;
        else:
            print "signal";
            misClassified += 1;
          
        avgPos = valPos/len(posList);
        avgNeg = valNeg/len(negList);
        avgPosValues.append(avgPos);
        avgNegValues.append(avgNeg);
        if avgPos < avgNeg:
            avgClass = 1;
            avgClassified += 1;
        else:
            avgClass = 0;
            avgMisClassified += 1; 
    f = open("results.txt","a");
    classification_rate = classified * 100/(classified+misClassified);
    f.write("MAX METHOD: classification rate "+str(classification_rate)+" classified: " +str(classified)+" misClassified: "+str(misClassified)+"\n");
    avgClassRate = avgClassified * 100/(avgClassified+avgMisClassified);
    f.write("AVERAGE METHOD: classification rate"+str(avgClassRate)+" classified: "+str(avgClassified)+" misClassified: "+str(avgMisClassified)+ "\n");    
    f.close();

shuffle(posSamples);
shuffle(negSamples);
per = 0.75

totalSamples = len(posSamples) + len(negSamples);
train_len = int(math.floor((per * totalSamples)/2));
#train_len = 5;
trainPos = posSamples[0:train_len];
testPos = posSamples[train_len:len(posSamples)];

trainNeg = negSamples[0:train_len];
testNeg = negSamples[train_len:len(negSamples)];
train_len = 3;    

(posList,negList) = trainData(trainPos,trainNeg);
testData(testPos,testNeg);

#conn = sqlite3.connect("project_db")
#c = conn.cursor()

#for samples in posSamples:
#    hmmObj = copy.deepcopy(hmm.HMM());
#    hmmObj.converge(samples,500);
#    posList.append(hmmObj);
#    print len(samples);
#
#for samples in negSamples:
#    hmmObj = copy.deepcopy(hmm.HMM());
#    hmmObj.converge(samples,500);
#    negList.append(hmmObj);
#    print len(samples);




#sam1 = posSamples[0];
#sam2 = posSamples[1000];
#sam3 = negSamples[500];

#sam1Obj = copy.deepcopy(hmm.HMM());
#sam1Obj.converge(sam1,300);
#posList.append(sam1Obj);
#sam2Obj = copy.deepcopy(hmm.HMM());
#A = [[0.4, 0.2, 0.2, 0.2],[0.2, 0.4, 0.2, 0.2],[ 0.2, 0.2, 0.4, 0.2],[ 0.2, 0.2, 0.2, 0.4]];
#B = [[0.4, 0.2, 0.2, 0.2],[ 0.2, 0.4, 0.2, 0.2],[ 0.2, 0.2, 0.4, 0.2],[ 0.2, 0.2, 0.2, 0.4] ];
#pi = [0.241896, 0.266086, 0.249153, 0.242864 ];
#A = np.array(A);
#B = np.array(B);
#pi = np.array(pi);

#sam1 = [0,1,2,3,3,0,0,1,1,1,2,2,2,3,0,0,0,1,1,1,2,3,3,0,0,0,1,1,1,2,3,3,0,1,2,3,0,1,1,1,2,3,3,0,1,2,2,3,0,0,0,1,1,2,2,3,0,1,1,2,3,0,1,2,2,2,2,3,0,0,1,2,3,0,1,1,2,3,3,3,0,0,1,1,1,1,2,2,3,3,3,0,1,2,3,3,3,3,0,1,1,2,2,3,0,0,0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,3,3,3,3,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,0,1,2,3,0,1,1,1,2,3,0,1,1,2,2,2,2,2,3,0,1,1,1,2,2,2,2,3,0,0,0,0,0,1,1,1,1,2,2,3,3,0,1,2,3,3,0,0,0,0,0,0,1,1,2,2,3,0,0,1,1,1,1,1,1,2,3,3,0,0,1,1,1,2,3,0,0,1,2,3,0,1,1,2,3,3,0,0,0,1,2,3,3,3,0,1,1,1,1,2,3,3,3,3,3,3,0,1,2,2,2,2,2,2,3,0,1,1,1,2,2,3,3,3,3,0,1,2,3,0,0,0,1,1,2,2,3,0,0,0,0,0,0,0,1,2,2,2,3,3,3,3,0,0,1,2,2,2,3,3,3,0,0,1,2,2,3,0,0,0,0,1,1,1,2,3,3,3,3,3,3,3,3,0,1,2,3,0,0,1,2,3,3,3,0,0,0,0,0,1,1,1,1,2,3,0,0,0,1,2,2,3,3,0,0,0,1,1,1,1,1,2,3,3,3,3,0,1,1,1,2,2,3,0,1,2,3,3,3,3,0,0,0,0,1,2,3,3,0,1,2,2,3,3,0,0,1,1,2,3,3,0,1,2,2,3,3,3,0,0,1,1,2,3,3,3,3,0,0,1,1,2,3,3,0,1,2,3,0,1,1,2,2,3,0,1,2,3,3,0,1,1,1,2,2,2,3,3,0,0,1,1,1,1,1,2,3,3,3,0,1,1,2,2,2,2,3,3,0,0,1,2,3,0,1,1,2,2,2,2,3,0,0,1,2,2,3,0,0,0,0,0,1,1,1,2,3,0,0,1,2,3,3,0,0,0,1,2,2,2,3,3,0,0,0,1,2,2,2,2,2,3,0,1,1,2,3,0,0,1,1,1,2,2,3,0,0,0,0,1,1,1,2,2,3,0,1,1,1,2,2,2,3,3,0,0,1,2,2,3,3,3,0,1,1,2,3,0,0,0,0,0,1,2,2,2,3,3,3,0,0,0,1,2,3,0,1,1,2,3,3,3,0,1,2,2,2,3,0,0,1,1,1,1,2,3,3,0,0,0,0,1,2,3,3,3,0,0,0,1,1,2,3,0,1,1,1,1,2,2,2,2,2,2,3,0,0,0,0,1,2,2,2,2,3,0,1,2,2,3,0,1,2,3,0,1,2,3,0,0,0,1,1,2,2,3,3,0,1,1,1,1,2,2,3,3,0,1,1,1,2,2,2,3,3,3,0,1,1,2,3,3,0,1,2,3,0,0,0,0,1,2,3,0,0,0,0,0,0,1,2,2,3,3,0,0,1,2,3,0,1,2,2,3,0,0,0,1,1,2,2,2,2,2,3,3,3,3,3,0,1,2,2,3,3,3,3,3,0,0,1,1,2,2,3,0,0,1,2,2,3,3,3,0,0,0,1,2,2,2,2,3,3,0,1,2,3,0,0,1,1,1,2,2,3,0,0,1,1,2,2,2,3,3,0,0,1,1,1,1,1,2,3,3,3,0,1,2,2,2,2,3,3,3,3,3,3,0,0,0,0,0,0,1,2,3,0,0,1,1,1,2,3,0,0,1,1,2,2,2,2,3,3,3,0,1,1,2,2,2,3,3,0,0,0,0,0,0,1,2,2,3,3,0,0,0,0,0,0,1,2,3,3,3,0,1,1,1,2,2,2,2,2,3,3,3,0,1,2,2,2,3,3,3,3,0,0,0,0,1,2,3,3,3,3,3,3,0,0,1,1,1,1,2,3,0,1,2,3,0,1,1,2,3,3,3,0,0,0,0,1,1,2,3,3,3,3,0,0,1,1,1,2,2,2,2,2,2,3,3,0,0,0,1,2,3,0,0,1,1,2,2,3,3,3,3,3,0,0,1,2,2,2,2,3,0,0,1,1,1,1,1,2,3,3,0,0,1,1,1,2,3,3,3,0,0];
#sam1Obj.setMatrices(A,B,pi);

#sam2Obj.converge(sam2,200);

#posList.append(sam2Obj);

#print posList[0].getMatrices();

#print posList[1].getMatrices();
#print "gamma";
#print sam1Obj.gamma;
#print sam2Obj.getMatrices();
#s = '';
#s1 = '';
#for i in sam1:
#    s += str(i);
#for i in sam2:
#    s1 += str(i);
#print s;
#print s1;

    
