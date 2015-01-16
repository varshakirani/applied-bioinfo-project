# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 21:01:53 2015

@author: varsha
"""
import numpy as np;
import math;
import random as rand;

class HMM(object):
#    N = 5;          #number of states (-n,-c,-h regions)
#    M = 21;         #number of observations(ACDEFGHIKLMNPQRSTVWY) 
                    #The above mentioned observations are mapped to integers from 1 to 20
    N = 4;
    M = 21;
    A = np.zeros((N,N));
    B = np.zeros((N,M));  
    pi = np.zeros(N);
    c = np.zeros(N);
    alpha = [];  #TxN
    beta = [];   #TxN
    digamma = []; #TxNxN
    gamma = []; #TxNxN
    
    
    def getMatrices(self):
        return (self.A,self.B,self.pi);

    def setMatrices(self,A_new,B_new,pi_new):
        self.A = A_new;
        self.B = B_new;
        self.pi = pi_new;
    
    def getAlpha(self):
        return self.alpha;

    def getBeta(self):
        return self.beta;

    def getGamma(self):
        return self.gamma;

    def getDiGamma(self):
        return self.digamma;

    def setAlpha(self,alpha_new):
        self.alpha = alpha_new;

    def setBeta(self,beta_new):
        self.beta = beta_new;

    def setGamma(self,gamma_new):
        self.gamma = gamma_new;

    def setDigamma(self,digamma_new):
        self.digamma = digamma_new;

    def forwardAlgorithm(self,obs_seq):
        T = len(obs_seq);
        alpha_tmp = np.zeros((T,self.N));
        
        for i in range(self.N):
            alpha_tmp[0][i] = self.pi[i] * self.B[i][obs_seq[0]];
        
        for t in range(1,T):
            for i in range(self.N):
                alpha_tmp[t][i] = 0;
                for j in range(self.N):
                    alpha_tmp[t][i] = alpha_tmp[t][i] + alpha_tmp[t-1][j] * self.A[j][i];
                alpha_tmp[t][i] = alpha_tmp[t][i] * self.B[i][obs_seq[t]];
                  
        prob_observation = 0;
        for i in range(self.N):
            prob_observation = prob_observation + alpha_tmp[T-1][i];
        
        return prob_observation;
    
    def alphaPass(self,obs_seq):
        T = len(obs_seq);
        c = np.zeros(T);
        A = self.A;
        B = self.B;
        pi = self.pi;
        N = self.N;
        alpha = np.zeros((T,N));  
#        print "A: alpha";
#        print self.A;
#        
#        print "B: alpha";
#        print self.B;
        for i in range(N):
            alpha[0][i] = pi[i] * B[i][obs_seq[0]];
            c[0] = c[0] + alpha[0][i];
#        if c[0] > 0:
        c[0] = 1 / c[0];
        
        for i in range(N):
            alpha[0][i] = c[0] * alpha[0][i];
            
        for t in range(1,T):
            c[t] = 0;
            for i in range(N):
                alpha[t][i] = 0;
                for j in range(N):
                    alpha[t][i] = alpha[t][i] + alpha[t-1][j] * A[j][i]; 
                alpha[t][i] = alpha[t][i] * B[i][obs_seq[t]];
                
                c[t] = c[t] + alpha[t][i];
#            if c[t] > 0:
            c[t] = 1 / c[t];
            
            for i in range(N):
                alpha[t][i] = c[t] * alpha[t][i];
                
        self.alpha = alpha;
        self.c = c;
#        print "scaling vector: alpha";
#        print self.c;
#        
#        print "alpha:";
#        print self.alpha;
#        

#        print "beta: alpha fun:";
#        print self.beta;
#        print "alpha:alpha fun:";
#        print self.alpha;
        return alpha;
        
    def betaPass(self,obs_seq):
        c = self.c;
        A = self.A;
        B = self.B;
        pi = self.pi;
        alpha = self.alpha;
        N = self.N;
        T = len(obs_seq);
        beta = np.zeros((T,N));
#        print "alpha: inside beta function";
#        print self.alpha;
        beta[0][0] = 1;
        for i in range(N):
            if T > 1:
                beta[T-1][i] = c[T-1];
                
        if T > 2:
            for t in range(T-2,-1,-1):
                for i in range(N):
                    beta[t][i] = 0;
                    for j in range(N):
                        beta[t][i] = beta[t][i] + A[i][j] * B[j][obs_seq[t+1]] * beta[t+1][j];
                    
                    beta[t][i] = c[t] * beta[t][i];
       
        
        self.beta = beta;
        
#        print "c:beta:";
#        print self.c;
#        
#        print "A: beta";
#        print self.A;
#        
#        print "B: beta";
#        print self.B;
#        print "beta:inside beta function: ";
#        print self.beta;
        return beta;
        
    def gammaDigamma(self,obs_seq):
        T = len(obs_seq);
        N = self.N;
        A = self.A;
        B = self.B;
        pi = self.pi;
        alpha = self.alpha;
        beta = self.beta;
        c = self.c;
        gamma = np.zeros((N,T));
        digamma = np.zeros((N,N,T));
        
        for t in range(T-1):
            denom = 0;
            for i in range(N):
                for j in range(N):
                    denom = denom + alpha[t][i] * A[i][j] * B[j][obs_seq[t+1]] * beta[t+1][j];
        
            for i in range(N):
                gamma[i][t] = 0;
                for j in range(N):
                    if denom > 0:
                        digamma[i][j][t] = (alpha[t][i] * A[i][j] * B[j][obs_seq[t+1]] * beta[t+1][j]) / denom ;
                        
                    gamma[i][t] = gamma[i][t] + digamma[i][j][t];
        self.gamma = gamma;
        self.digamma = digamma;
#        print "c:gammadigamma:";
#        print self.c;
#        print "gamma";
#        print self.gamma;
#        print "digamma";
#        print self.digamma;
        return (self.gamma,self.digamma);
        
    def reestimate(self,obs_seq):
        A = self.A;
        B = self.B;
        pi = self.pi;
        gamma = self.gamma;
        digamma = self.digamma;
        alpha = self.alpha;
        beta = self.beta;
        c = self.c;
        N = self.N;
        M = self.M;
        
        
        T = len(obs_seq);
        
        #reestimate pi
        for i in range(N):
            pi[i] = gamma[i][0];
            
        #reestimate A
        for i in range(N):
            for j in range(N):
                numer = 0;
                denom = 0;
                for t in range(T-1):
                    numer = numer + digamma[i][j][t];
                    denom = denom + gamma[i][t];
                    
                
                A[i][j] = numer/denom;
                
        
        #reestimate B
        for i in range(N):
            for j in range(M):
                numer = 0;
                denom = 0;
#                print "observation"
#                print obs_seq;
                dummy = j;
                for t in range(T):
                    if obs_seq[t] == j:
                        
                        numer = numer + gamma[i][t];
                        
                    denom = denom + gamma[i][t];
                    
#                print "numer: ";
#                print numer;
            
                if numer == 0:
#                    print "gamma";
#                    print gamma[i];
#                    print "j:"
#                    print dummy
                    numer = 0.00000000001;
                B[i][j] = numer / denom;
               
        self.pi = pi;
        self.A = A;
        self.B = B;
#        print "c:reestimate:";
#        print self.c;
        return(self.pi,self.A,self.B);
        
    def logProbCal(self,obs_seq):
        logProb = 0;
        T = len(obs_seq);
        c = self.c;
        for t in range(T-1):
            logProb = logProb + math.log(c[t]);
            
        logProb = -logProb;
        
        return logProb;
        
    def converge(self,obs_seq,maxIters):
        iters = 0;
        oldLogProb = -9999999999999999999999999999999999999999;
        
        self.alphaPass(obs_seq);
        self.betaPass(obs_seq);
        self.gammaDigamma(obs_seq);
        self.reestimate(obs_seq);
        logProb = self.logProbCal(obs_seq);
        print "logProb: ";
        print logProb;        
        iters = iters + 1;
        
        while iters<maxIters and (logProb - oldLogProb) >  math.pow(10,-6):
            self.alphaPass(obs_seq);
            self.betaPass(obs_seq);
            self.gammaDigamma(obs_seq);
            self.reestimate(obs_seq);
            oldLogProb = logProb;
            logProb = self.logProbCal(obs_seq);
            print logProb;            
            iters = iters + 1;
        print logProb;
        print iters
#        print self.A
        
                
        
    def __init__(self):
        self.initializeMatrices();
    def initializeMatrices(self):
        noise = np.zeros(self.N);
        noise[0]=0.6;
        noise[1]=0.2;
        noise[2]=0.3;
        noise[3]=0.15;
#        noise[4]=0.45;
#        noise[5]=0.89;
        N = self.N;
        M = self.M;
        A = self.A;
        B = self.B;
        pi = self.pi;
        for i in range(N):
            index = 0;
            val = 0;
            A[i][i] = 0.4;
            for j in range(N):
                if i is not j:
                    A[i][j] = 0.6 / 4;
                    A[i][j] = A[i][j] + noise[index];
                    val = val + A[i][j];
                    index = index + 1;
            for j in range(N):
                if i is not j :
                    A[i][j] = (A[i][j] / val)*0.6;
                    
                    
        #B initialization
        
        for i in range(N):
            index = 0;
            val = 0;
            B[i][i] = 0.15;
            for j in range(M):
                if i is not j:
                    B[i][j] = 0.85 / 4;
                    B[i][j] = B[i][j] +noise[rand.randint(0,self.N-1 )];
                    val = val+B[i][j]
                    index = index + 1;
                    
            for j in range(M):
                if i is not j:
                    B[i][j] = (B[i][j] / val)*0.85;
                    
        #pi initialization
        val = 0;
        for i in range(N):
            pi[i] = rand.random();
            val = val + pi[i];
        for i in range(N):
            pi[i] = pi[i] / val;
            
        self.A = A;
        self.B = B;
        self.pi = pi;
                

#end of HMM class                


#hmmObject = HMM();

#To check forward algorithm#                
A = [[0.0, 0.8, 0.1, 0.1],[0.1, 0.0, 0.8, 0.1],[0.1, 0.1, 0.0, 0.8],[0.8, 0.1, 0.1, 0.0] ];
A = np.array(A);
 
B = [[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1],[ 0.1, 0.0, 0.0, 0.9]];
B = np.array(B);
  
pi = [1.0 ,0.0 ,0.0 ,0.0 ];
pi = np.array(pi);
 
obs_seq = [0, 1, 2, 3, 0, 1, 2, 3 ];
obs_seq = np.array(obs_seq); 
#hmmObject.setMatrices(A,B,pi);

#hmmObject.initializeMatrices();
#(a,b,p) = hmmObject.getMatrices();
#print "before convergence:";
#print "A:";
#print a;
#print "B:";
#print b;
#print "pi:";
#print p
#hmmObject.converge(obs_seq,300);
#
#(a,b,p) = hmmObject.getMatrices();
#print "after convergence:";
#print "A:";
#print a;
#print "B:";
#print b;
#print "pi:";
#print p

#print "Probability:";
#print hmmObject.forwardAlgorithm(obs_seq);
#hmmObject.alphaPass(obs_seq);
#hmmObject.betaPass(obs_seq);
#(gamma,digamma) = hmmObject.gammaDigamma(obs_seq);
#print gamma;
#print digamma;
#(pi,A,B) = hmmObject.reestimate(obs_seq);
#print pi;
#print A;
#print B;
#end of checking forward algorithm


# to check Hmm model parameter estimation
### first input
#A = [[0.4, 0.2, 0.2, 0.2],[0.2, 0.4, 0.2, 0.2],[ 0.2, 0.2, 0.4, 0.2],[ 0.2, 0.2, 0.2, 0.4]];
#B = [[0.4, 0.2, 0.2, 0.2],[ 0.2, 0.4, 0.2, 0.2],[ 0.2, 0.2, 0.4, 0.2],[ 0.2, 0.2, 0.2, 0.4] ];
#pi = [0.241896, 0.266086, 0.249153, 0.242864 ];
#
#obs_seq = [0,1,2,3,3,0,0,1,1,1,2,2,2,3,0,0,0,1,1,1,2,3,3,0,0,0,1,1,1,2,3,3,0,1,2,3,0,1,1,1,2,3,3,0,1,2,2,3,0,0,0,1,1,2,2,3,0,1,1,2,3,0,1,2,2,2,2,3,0,0,1,2,3,0,1,1,2,3,3,3,0,0,1,1,1,1,2,2,3,3,3,0,1,2,3,3,3,3,0,1,1,2,2,3,0,0,0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,3,3,3,3,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,0,1,2,3,0,1,1,1,2,3,0,1,1,2,2,2,2,2,3,0,1,1,1,2,2,2,2,3,0,0,0,0,0,1,1,1,1,2,2,3,3,0,1,2,3,3,0,0,0,0,0,0,1,1,2,2,3,0,0,1,1,1,1,1,1,2,3,3,0,0,1,1,1,2,3,0,0,1,2,3,0,1,1,2,3,3,0,0,0,1,2,3,3,3,0,1,1,1,1,2,3,3,3,3,3,3,0,1,2,2,2,2,2,2,3,0,1,1,1,2,2,3,3,3,3,0,1,2,3,0,0,0,1,1,2,2,3,0,0,0,0,0,0,0,1,2,2,2,3,3,3,3,0,0,1,2,2,2,3,3,3,0,0,1,2,2,3,0,0,0,0,1,1,1,2,3,3,3,3,3,3,3,3,0,1,2,3,0,0,1,2,3,3,3,0,0,0,0,0,1,1,1,1,2,3,0,0,0,1,2,2,3,3,0,0,0,1,1,1,1,1,2,3,3,3,3,0,1,1,1,2,2,3,0,1,2,3,3,3,3,0,0,0,0,1,2,3,3,0,1,2,2,3,3,0,0,1,1,2,3,3,0,1,2,2,3,3,3,0,0,1,1,2,3,3,3,3,0,0,1,1,2,3,3,0,1,2,3,0,1,1,2,2,3,0,1,2,3,3,0,1,1,1,2,2,2,3,3,0,0,1,1,1,1,1,2,3,3,3,0,1,1,2,2,2,2,3,3,0,0,1,2,3,0,1,1,2,2,2,2,3,0,0,1,2,2,3,0,0,0,0,0,1,1,1,2,3,0,0,1,2,3,3,0,0,0,1,2,2,2,3,3,0,0,0,1,2,2,2,2,2,3,0,1,1,2,3,0,0,1,1,1,2,2,3,0,0,0,0,1,1,1,2,2,3,0,1,1,1,2,2,2,3,3,0,0,1,2,2,3,3,3,0,1,1,2,3,0,0,0,0,0,1,2,2,2,3,3,3,0,0,0,1,2,3,0,1,1,2,3,3,3,0,1,2,2,2,3,0,0,1,1,1,1,2,3,3,0,0,0,0,1,2,3,3,3,0,0,0,1,1,2,3,0,1,1,1,1,2,2,2,2,2,2,3,0,0,0,0,1,2,2,2,2,3,0,1,2,2,3,0,1,2,3,0,1,2,3,0,0,0,1,1,2,2,3,3,0,1,1,1,1,2,2,3,3,0,1,1,1,2,2,2,3,3,3,0,1,1,2,3,3,0,1,2,3,0,0,0,0,1,2,3,0,0,0,0,0,0,1,2,2,3,3,0,0,1,2,3,0,1,2,2,3,0,0,0,1,1,2,2,2,2,2,3,3,3,3,3,0,1,2,2,3,3,3,3,3,0,0,1,1,2,2,3,0,0,1,2,2,3,3,3,0,0,0,1,2,2,2,2,3,3,0,1,2,3,0,0,1,1,1,2,2,3,0,0,1,1,2,2,2,3,3,0,0,1,1,1,1,1,2,3,3,3,0,1,2,2,2,2,3,3,3,3,3,3,0,0,0,0,0,0,1,2,3,0,0,1,1,1,2,3,0,0,1,1,2,2,2,2,3,3,3,0,1,1,2,2,2,3,3,0,0,0,0,0,0,1,2,2,3,3,0,0,0,0,0,0,1,2,3,3,3,0,1,1,1,2,2,2,2,2,3,3,3,0,1,2,2,2,3,3,3,3,0,0,0,0,1,2,3,3,3,3,3,3,0,0,1,1,1,1,2,3,0,1,2,3,0,1,1,2,3,3,3,0,0,0,0,1,1,2,3,3,3,3,0,0,1,1,1,2,2,2,2,2,2,3,3,0,0,0,1,2,3,0,0,1,1,2,2,3,3,3,3,3,0,0,1,2,2,2,2,3,0,0,1,1,1,1,1,2,3,3,0,0,1,1,1,2,3,3,3,0,0];
##obs_seq = [10, 8, 0, 8, 8, 20, 3, 11, 13, 7, 20, 11, 3, 11, 5, 14, 14, 1, 13, 14, 6, 5, 14, 14, 9, 0, 7, 0, 2];

####
#hmmObject.setMatrices(A,B,pi);
#(a,b,p) = hmmObject.getMatrices();
#print "before convergence:";
#print "A:";
#print a;
#print "B:";
#print b;
#print "pi:";
#print p
#hmmObject.converge(obs_seq,25);
#
#(a,b,p) = hmmObject.getMatrices();
#print "after convergence:";
#print "A:";
#print a;
#print "B:";
#print b;
#print "pi:";
#print p

#Second Input

#A = [[0.2,0.4,0.2,0.2],[0.4,0.2,0.2,0.2],[0.2,0.2,0.2,0.4],[0.2,0.2,0.4,0.2]];
#B = [[0.7,0.1,0.1,0.1],[0.1,0.7,0.1,0.1],[0.1,0.1,0.7,0.1],[0.1,0.1,0.1,0.7]];
#pi = [1.0,0.0,0.0,0.0];
#
#obs_seq = [1,0,1,0,2,3,0,1,0,1,2,3,2,0,1,0,3,2,3,2,3,2,1,0,1,0,1,2,3,2,0,2,1,0,1,3,2,3,2,3,2,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,3,2,0,1,0,1,2,3,2,3,0,3,2,1,0,3,2,1,2,3,0,1,0,1,2,0,3,0,1,0,3,0,1,2,3,1,2,3,2,1,0,1,0,3,2,3,2,1,2,1,0,2,1,3,2,3,2,3,1,0,1,0,3,2,3,2,0,1,0,2,1,0,1,0,3,1,0,1,0,2,1,0,3,2,3,2,3,2,3,0,1,2,1,0,1,0,1,0,3,2,3,2,0,1,3,2,3,2,3,0,1,0,3,2,3,2,3,0,1,3,2,3,1,0,1,0,3,1,0,1,2,3,0,1,0,3,2,3,0,1,0,1,2,1,0,3,0,1,0,1,3,2,3,1,0,1,2,1,0,3,2,1,0,2,3,1,2,3,2,3,2,3,2,3,0,3,0,1,2,3,2,1,3,2,0,3,2,3,2,3,2,1,0,1,3,0,1,3,2,3,2,0,3,2,3,2,3,2,3,0,1,0,1,2,3,2,3,2,1,0,1,0,3,0,1,0,2,3,0,1,0,1,0,1,2,1,0,1,0,1,3,2,3,2,3,2,3,0,2,1,2,3,2,3,2,0,1,0,1,0,1,0,3,2,3,2,3,0,1,0,1,0,2,3,2,3,2,3,2,3,2,1,0,2,3,2,3,0,1,2,3,2,1,0,1,2,0,3,1,0,1,0,3,0,2,3,1,0,1,2,1,2,3,0,3,0,1,0,3,2,1,0,1,0,3,2,0,1,0,1,2,3,0,3,2,1,0,3,2,0,1,0,1,0,1,0,1,0,1,2,0,1,3,2,3,2,3,0,2,1,0,3,2,1,0,1,0,1,0,2,3,2,1,0,1,0,1,0,1,2,3,0,1,2,0,1,0,1,0,1,0,1,0,3,2,1,0,1,0,1,0,1,2,3,2,3,0,1,0,1,0,3,2,3,2,1,2,1,0,1,0,1,2,3,2,3,2,3,2,3,0,3,0,1,0,3,2,3,2,3,2,1,2,3,2,3,2,3,0,1,3,0,2,1,0,1,0,1,3,0,1,0,1,0,1,0,3,2,3,2,1,0,1,3,1,0,1,0,3,0,1,0,2,3,2,3,2,3,2,3,2,1,2,3,2,3,2,1,0,1,0,1,2,3,2,3,2,1,0,1,2,1,0,1,3,0,3,2,3,2,3,0,1,0,1,2,3,2,1,0,1,0,1,3,2,1,0,3,2,1,0,3,2,3,2,0,1,0,1,2,3,0,2,3,2,3,2,1,0,1,2,3,2,1,0,1,3,2,0,1,2,3,0,2,3,2,3,0,1,0,1,2,3,2,0,1,0,3,2,0,1,0,1,0,1,0,1,0,1,0,1,3,2,3,2,1,2,3,2,1,0,1,0,1,0,1,0,3,0,1,0,1,0,3,0,1,0,1,2,3,0,1,0,3,0,1,0,1,0,1,2,3,0,1,0,3,0,1,0,1,2,3,0,1,0,1,2,1,3,0,1,0,1,0,1,0,3,2,3,2,3,2,1,0,1,0,1,3,0,3,0,1,2,3,1,3,2,3,2,1,3,1,2,1,0,1,0,1,0,2,1,0,1,0,1,2,3,2,1,0,2,3,2,3,2,3,2,3,0,3,2,3,0,1,2,3,2,3,2,3,2,3,1,0,1,0,1,0,1,2,0,3,0,3,2,0,3,2,1,0,1,0,1,0,1,2,3,2,3,2,1,2,3,0,1,0,3,2,0,3,2,0,1,0,1,0,3,0,1,0,1,2,3,2,3,2,1,0,1,0,1,2,3,2,3,2,3,2,1,0,1,0,1,0,3,0,1,2,1,0,3,2,3,2,3,2,0,3,2,3,2,3,0,1,2,3,1,0,1,0,3,2,3,2,1,2,3,0,3,2,3,2,3,2,1,2,3,2,3,2,1,2,3,2,3,2,3,2,1,0,1,0,1,0,3,2,3,0,3,2,3,2,3,0,1,2,1,0,3,0,3,2,3,2,3,2,0,1,2,3,0,1,0,1,2,0,1,0,1,0,1,0,1,3,2,1,0,1,0,1,0,1,0,3,2,3,2,1,0,1,0,1,0,3,2,1,0,1,0,1,0,1,3,2,3,0,3,0,1];

# thirs sample##

A = [[0.7,0.1,0.1,0.1],[0.1,0.7,0.1,0.1],[0.1,0.1,0.7,0.1],[0.1,0.1,0.1,0.7]];
B = [[0.4,0.2,0.2,0.2],[0.2,0.4,0.2,0.2],[0.2,0.2,0.4,0.2],[0.2,0.2,0.2,0.4]];
B_new = [[0.4,0.2,0.2,0.1,0.1],[0.2,0.3,0.1,0.2,0.2],[0.2,0.1,0.2,0.1,0.3],[0.1,0.1,0.2,0.2,0.4]];

pi = [1.0,0.0,0.0,0.0];
obs_seq = [0,0,0,0,2,0,0,0,2,3,3,2,2,3,1,1,2,2,2,2,2,2,2,0,1,1,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,3,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,3,0,0,0,0,1,3,3,1,2,2,2,2,1,2,2,2,2,0,0,2,1,0,0,1,1,1,1,2,2,2,0,2,1,1,1,1,1,2,2,2,2,2,2,2,1,2,2,2,3,1,1,2,1,1,2,2,2,2,2,1,2,1,1,1,1,1,3,3,3,3,3,3,3,3,2,2,2,2,2,1,2,2,2,3,3,3,3,3,3,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,3,3,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,0,0,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,2,1,1,2,1,1,1,1,2,3,2,2,1,1,3,3,1,1,1,1,1,3,1,1,1,2,2,2,2,2,2,2,2,1,1,3,3,3,3,3,3,3,3,3,2,2,2,2,1,1,2,2,2,1,2,3,3,2,2,2,2,1,0,1,1,1,1,2,1,2,1,2,1,1,2,2,1,2,2,2,2,2,2,1,2,2,2,2,2,2,2,0,3,3,0,0,1,1,3,2,2,2,2,2,2,2,2,2,2,2,2,2,3,2,2,2,1,1,3,2,2,2,0,0,1,2,2,3,3,3,3,3,2,1,2,1,2,0,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,0,0,0,0,0,2,1,2,2,0,2,2,3,3,3,2,2,2,1,2,2,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,1,3,1,3,2,2,2,2,2,2,1,2,2,3,2,2,3,2,2,1,1,1,1,1,3,3,3,3,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,1,2,0,0,1,3,2,3,3,2,3,3,2,2,2,2,2,1,1,1,1,2,1,3,2,2,2,1,0,2,2,2,2,2,1,2,3,1,2,1,1,1,1,1,2,3,2,2,1,3,2,2,2,2,2,2,2,3,1,1,1,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,1,2,2,2,3,2,1,2,2,2,2,2,2,2,2,2,2,2,0,0,0,1,0,0,2,2,2,2,2,2,2,2,3,3,3,1,2,2,2,2,2,2,2,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,1,1,1,1,3,1,3,3,3,3,3,2,2,2,1,2,2,2,3,1,3,3,2,2,2,2,2,2,2,3,3,3,1,1,1,1,1,0,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,0,0,0,1,3,3,3,3,3,1,2,2,2,2,2,2,1,2,3,3,2,1,1,1,3,1,1,1,2,1,2,2,2,2,1,2,2,2,2,2,2,0,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,2,0,0,0,3,0,0,2,2,2,2,2,2,2,2,0,1,2,2,1,2,1,1,2,2,2,2,2,2,3,2,2,2,2,2,2,0,0,0,1,2,2,2,2,2,2,2,1,2,2,2,2,3,2,2,2,2,2,2,2,2,2,1,0,1,2,2,1,3,3,3,2,0,0,0,1,1,1,3,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,3,3,2,2,2,2,2,0,0,1,3,3,3,3,3,3,3,2,2,2,3,2,2,2,2,1,2,2,3,3,1,1,1,3,3,3,3,3,2,2,2,2,2,2,2,2,1,3,3,2,0,0,2,2,2,1,1,1,3,2,2,2,2,2,2,2,1,2,1,2,3,3,3,3,2,2,1,1,2,2,2,2,2,2,2,1,3,0,1,2,2,0,3,0,0,0,0,2,3,2,1,3];
obs_seq = [0,0,0,0,2,0,0,0,2,3,3,2,2,3,1,1,2,2,2,2,2,2,2,0,1,1,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,3,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,3,0,0,0,0,1,3,3,1,2,2,2,2,1,2,2,2,2,0,0,2,1,0,0,1,1,1,1,2,2,2,0,2,1,1,1,1,1,2,2,2,2,2,2,2,1,2,2,2,3,1,1,2,1,1,2,2,2,2,2,1,2,1,1,1,1,1,3,3,3,3,3,3,3,3,2,2,2,2,2,1,2,2,2,3,3,3,3,3,3,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,3,3,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,0,0,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,2,1,1,2,1,1,1,1,2,3,2,2,1,1,3,3,1,1,1,1,1,3,1,1,1,2,2,2,2,2,2,2,2,1,1,3,3,3,3,3,3,3,3,3,2,2,2,2,1,1,2,2,2,1,2,3,3,2,2,2,2,1,0,1,1,1,1,2,1,2,1,2,1,1,2,2,1,2,2,2,2,2,2,1,2,2,2,2,2,2,2,0,3,3,0,0,1,1,3,2,2,2,2,2,2,2,2,2,2,2,2,2,3,2,2,2,1,1,3,2,2,2,0,0,1,2,2,3,3,3,3,3,2,1,2,1,2,0,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,0,0,0,0,0,2,1,2,2,0,2,2,3,3,3,2,2,2,1,2,2,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,1,3,1,3,2,2,2,2,2,2,1,2,2,3,2,2,3,2,2,1,1,1,1,1,3,3,3,3,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,1,2,0,0,1,3,2,3,3,2,3,3,2,2,2,2,2,1,1,1,1,2,1,3,2,2,2,1,0,2,2,2,2,2,1,2,3,1,2,1,1,1,1,1,2,3,2,2,1,3,2,2,2,2,2,2,2,3,1,1,1,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,1,2,2,2,3,2,1,2,2,2,2,2,2,2,2,2,2,2,0,0,0,1,0,0,2,2,2,2,2,2,2,2,3,3,3,1,2,2,2,2,2,2,2,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,1,1,1,1,3,1,3,3,3,3,3,2,2,2,1,2,2,2,3,1,3,3,2,2,2,2,2,2,2,3,3,3,1,1,1,1,1,0,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,0,0,0,1,3,3,3,3,3,1,2,2,2,2,2,2,1,2,3,3,2,1,1,1,3,1,1,1,2,1,2,2,2,2,1,2,2,2,2,2,2,0,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,2,0,0,0,3,0,0,2,2,2,2,2,2,2,2,0,1,2,2,1,2,1,1,2,2,2,2,2,2,3,2,2,2,2,2,2,0,0,0,1,2,2,2,2,2,2,2,1,2,2,2,2,3,2,2,2,2,2,2,2,2,2,1,0,1,2,2,1,3,3,3,2,0,0,0,1,1,1,3,2,2,2,2,2,2,2,2,2,2,2,2,4,1,2,2,2,2,2,3,3,2,2,2,2,2,0,0,1,3,3,3,3,3,3,3,2,2,2,3,2,2,2,2,1,2,2,3,3,1,4,1,1,3,3,3,3,3,2,2,2,2,2,2,2,2,1,3,3,2,0,4,0,2,2,2,1,1,1,3,2,2,2,4,2,2,2,2,1,2,1,2,3,3,3,3,2,2,4,1,1,2,2,2,2,2,2,4,2,1,3,0,1,2,2,0,3,0,4,0,0,0,2,3,2,1,4];
A = np.array(A);
B = np.array(B);
B_new = np.array(B_new);
pi = np.array(pi);
obs_seq = np.array(obs_seq);
#obj = HMM();
#obj.setMatrices(A,B_new,pi);
#obj.converge(obs_seq,300);
#(a,b,p) = obj.getMatrices();
#print "after convergence:";
#print "A:";
#print a;
#print "B:";
#print b;
#print "pi:";
#print p