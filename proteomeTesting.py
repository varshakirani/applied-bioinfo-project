# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 23:12:59 2015

@author: varsha
"""

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
    
    letters = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','X','Y','U','*'];
    values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,19];
    dictionary = dict(zip(letters, values));
    obs_seq =np.zeros(len(seq));
    index = 0;    
    
    for alphabet in seq:
        obs_seq[index] = dictionary[alphabet];
        index += 1;    
    
    obs_seq = obs_seq.astype(int);
    return obs_seq.tolist();
    
path = "../../data/testProteome";
drosophilaSamples = [];
homoSapiensSamples = [];
musculusSamples = [];
sacchSamples = [];
for root,dirs,files in os.walk(path):
     for names in files:
         if names.startswith(("Dro")):
             handle = open(str(root)+"/"+str(names),"rU");
             for record in SeqIO.parse(handle,"fasta"):   #reads FASTA file and record will have id and sequence
                 s = str(record.seq)
                 if len(s) > 40:
                     signalPeptide = s[0:40];
                 else:
                     signalPeptide = s[0:len(s)];
                 
                 obs_seq = convertToNumArr(signalPeptide);
                 drosophilaSamples.append(obs_seq);
             handle.close()
         elif names.startswith(("Homo")):
             handle = open(str(root)+"/"+str(names),"rU");
             for record in SeqIO.parse(handle,"fasta"):   #reads FASTA file and record will have id and sequence
                 s = str(record.seq)
                 if len(s) > 40:
                     signalPeptide = s[0:40];
                 else:
                     signalPeptide = s[0:len(s)];
                 obs_seq = convertToNumArr(signalPeptide);
                 homoSapiensSamples.append(obs_seq);
             handle.close()
         elif names.startswith(("Mus")):
             handle = open(str(root)+"/"+str(names),"rU");
             for record in SeqIO.parse(handle,"fasta"):   #reads FASTA file and record will have id and sequence
                 s = str(record.seq)
                 if len(s) > 40:
                     signalPeptide = s[0:40];
                 else:
                     signalPeptide = s[0:len(s)];
                 print signalPeptide
                 obs_seq = convertToNumArr(signalPeptide);
                 musculusSamples.append(obs_seq);
             handle.close()
         elif names.startswith(("Sacc")):
             handle = open(str(root)+"/"+str(names),"rU");
             for record in SeqIO.parse(handle,"fasta"):   #reads FASTA file and record will have id and sequence
                 s = str(record.seq)
                 if len(s) > 40:
                     signalPeptide = s[0:40];
                 else:
                     signalPeptide = s[0:len(s)];
                 obs_seq = convertToNumArr(signalPeptide);
                 sacchSamples.append(obs_seq);
             handle.close()             
             
                    
def getSamples():
    return (drosophilaSamples,homoSapiensSamples,musculusSamples,sacchSamples);
    