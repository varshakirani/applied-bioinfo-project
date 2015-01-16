# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:03:14 2015

@author: varsha
"""

import signalPeptide as sp;


#print sp.posList[0].A;

posFile = open('outputPosFile.txt','w');
for obj in sp.posList:
    strA = '';
    strB = '';
    strPi = '';
    rowA = len(obj.A);
    colA = len(obj.A[0]);

    rowB = len(obj.B);
    colB = len(obj.B[0]);    

    colPi = len(obj.pi);
    strA += str(rowA) + ' ' + str(colA) + ' ';
    strB += str(rowB) + ' ' + str(colB) + ' ';
    strPi += '1' + ' ' + str(colPi) + ' ';
    for row in obj.A:
        for col in row:
            strA += str(col) + ' ';            
            
    for row in obj.B:
        for col in row:
            strB += str(col) + ' ';       
    
    for col in obj.pi:
        strPi += str(col) + ' ';
    
    posFile.write(strA + '\n');
    posFile.write(strB + '\n');
    posFile.write(strPi + '\n');
#    posFile.write(obj.A);
#    posFile.write(obj.B);
#    posFile.write(obj.pi);
#    posFile.write('\n');