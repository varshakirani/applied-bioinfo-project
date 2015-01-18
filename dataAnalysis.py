# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 00:28:44 2015

@author: varsha
"""

import inputRead as ir;


def writeSamplesIntoFiles(pos,neg):
    fp = open("posFileData","w");
    fn = open("negFileData","w");
        
    for samp in pos:
        s = '';
        for ltr in samp:
            s += str(ltr+1) + ' ';
        s += "\n";
        fp.write(s);

    for samp in neg:
        s = '';
        for ltr in samp:
            s += str(ltr+1) + ' ';
        s += "\n";
        fn.write(s);
        

(posSamples,negSamples)= ir.getSamples();
writeSamplesIntoFiles(posSamples,negSamples);