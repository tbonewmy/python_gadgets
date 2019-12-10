# -*- coding: utf-8 -*-
"""
* Copyright (c) 2019 Tallahassee, Mingyuan Wang
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
discretize feature values according to its sample distribution in both class labels
"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


class dist_cate():
    def __init__(self,threshold=0.004,est_point=100,skip_num=1,draw=False,save_path=''):
        #threshold: threshold to control number of splits
        #est_point: number of point used to eatimate pdf
        #skip_num: number of skipped pdf point to calculate intersections
        self.threshold=threshold
        self.est_point=est_point
        self.skip_num=skip_num
        self.draw=draw
        if draw:
            if len(save_path)==0:
                raise Exception('needs image save path')           
            self.save_path=save_path
            
        
    def get_edges(self,X1,X0):
        #get min and max
        self.min_v1=X1.min(axis=0).values
        self.min_v0=X0.min(axis=0).values
        self.max_v1=X1.max(axis=0).values
        self.max_v0=X0.max(axis=0).values
        self.edges=[]
        if self.draw:
            f,ax=plt.subplots(int(np.ceil(self.M/5)),5,figsize=(5*5,np.ceil(self.M/5)*5))
        for i,c in enumerate(self.col_names):
            start=np.max([self.min_v1[i],self.min_v0[i]])
            finish=np.min([self.max_v1[i],self.max_v0[i]])
            x_line=np.linspace(start,finish,self.est_point)
            g1=gaussian_kde(X1[c])
            pdf1=g1.evaluate(x_line)
            g0=gaussian_kde(X0[c])
            pdf0=g0.evaluate(x_line)
            edge=[]
            height=np.mean([np.max(pdf1),np.max(pdf0)])
            max_dis=height*self.threshold
            for j in range(len(pdf1)-self.skip_num):
                mag=(np.abs(pdf1[j]-pdf0[j])+np.abs(pdf1[j+self.skip_num]-pdf0[j+self.skip_num]))/2>max_dis
                if pdf1[j]>pdf0[j]:
                    if pdf1[j+self.skip_num]<pdf0[j+self.skip_num] and mag:
                        edge.extend([(x_line[j+self.skip_num]+x_line[j])/2])
                else:
                    if pdf1[j+self.skip_num]>pdf0[j+self.skip_num] and mag:
                        edge.extend([(x_line[j+self.skip_num]+x_line[j])/2])
            self.edges.append(edge)
            if self.draw:
                ax[i//5,i%5].plot(x_line,pdf1,label='class1')
                ax[i//5,i%5].plot(x_line,pdf0,label='class0')
                ax[i//5,i%5].set_title(c)
                ax[i//5,i%5].legend(loc='best')
                for e in edge:
                    ax[i//5,i%5].axvline(e)
        if self.draw:
            plt.tight_layout()
            plt.savefig(self.save_path)
 
    def fit(self,X,y):
        #X pandas Dataframe, y Series
        #split data
        X1=X[y==1]
        X0=X[y==0]
        self.N,self.M=X.shape
        self.col_names=X1.columns
        
        self.get_edges(X1,X0)
        
    def cut_bins(self,col,edge):
        le=len(edge)
        if le==0:
            new_col=col
        else:
            new_col=np.zeros(self.N)
            for i in range(le):
               new_col[col>edge[i]] += 1
        return new_col
        
    def transform(self,X):
        for i,c in enumerate(self.col_names):
            X[c]=self.cut_bins(X[c].values,self.edges[i])      
        return X

    def fit_transform(self,X,y):
        self.fit(X,y)
        X=self.transform(X)
        return X
    
#    def drawplot(self,col_name):
#        plt.plot(self.x_line,p1)
#        plt.plot(self.x_line,p0)
#        for i in inter:
#            plt.axvline(i)