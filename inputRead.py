# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 20:02:47 2015

@author: varsha
"""
import os;
from Bio import SeqIO;
from Bio.Seq import Seq;
import numpy as np;



def convertToNumArr(seq):
    
    letters = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','X','Y'];
    values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
    dictionary = dict(zip(letters, values));
    obs_seq =np.zeros(len(seq));
    index = 0;    
    
    for alphabet in seq:
        obs_seq[index] = dictionary[alphabet];
        index += 1;    
    
    obs_seq = obs_seq.astype(int);
    return obs_seq.tolist();
    
path = "../../data/training_data/positive_examples";
posSamples = [];
for root,dirs,files in os.walk(path):
     for names in files:
         if names.endswith((".faa")) and not names.startswith((".")):
#             print names;
#             print root;
#             print dirs;
             handle = open(str(root)+"/"+str(names),"rU");
             for record in SeqIO.parse(handle,"fasta"):   #reads FASTA file and record will have id and sequence
                 s = str(record.seq)
#                 print s
                 annotationIndex = s.index('#');
                 annotation = s[annotationIndex:];
#                 print annotation;
                 lenSignalPeptide = annotation.index('C');
#                 print lenSignalPeptide;
                 
                 signalPeptide = s[0:lenSignalPeptide];
#                 print signalPeptide;
                 obs_seq = convertToNumArr(signalPeptide);
                 posSamples.append(obs_seq);
             handle.close()
    
print posSamples[0];
            
negSamples = [];
negPath = "../../data/training_data/negative_examples";
for root,dirs,files in os.walk(negPath):
    for names in files:
        if names.endswith((".faa")) and not names.startswith((".")):
            handle = open(str(root)+"/"+str(names),"rU");
            for record in SeqIO.parse(handle,"fasta"):
                s = str(record.seq);
                annotationIndex = s.index('#');
                maxLen = 25;
                if annotationIndex < maxLen:
                    maxLen = annotationIndex
                s = s[0:maxLen];
                obs_seq = convertToNumArr(s);
                negSamples.append(obs_seq);
#                for extensive negative samples
#                for i in range(5,30):
#                    sample = s[0:i];
#                    obs_seq = convertToNumArr(sample);
#                    negSamples.append(obs_seq);
                    
def getSamples():
    return (posSamples,negSamples);
    