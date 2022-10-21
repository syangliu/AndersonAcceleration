import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

global colors, linestyles, markers
colors = [(0,128/255,1), (0,128/255,1), (1,128/255, 0), (1,128/255, 0), (152/255,24/255,147/255), (1,0,0), (0,0,0), 'b', 'c']
linestyles = ['-', '--', '-', '--', '-', '-', '-', '--', '-.']
markers = ['*', '*', '*', 'o',  'X', 'X', 'P', 'P', 'X', '1', '2', '.', 'o']


def draw(plt_fun, record, label, i, NC, yaxis, xaxis=None):
    if not (xaxis is not None):
        xaxis = torch.tensor(range(1,len(yaxis)+1))
    plt_fun(xaxis, yaxis, color=colors[i], 
            linestyle=linestyles[i], label = label)
    if NC:
        index = (record[:,5][1:] == True)
        if xaxis is not None:
            xNC = xaxis[:-1][index]
        else:
            xNC = torch.tensor(range(1,len(yaxis)))[index]
        yNC = yaxis[:-1][index]
        plt_fun(xNC, yNC, '.', color=colors[i], marker=markers[0], markersize=8)
        
def showFigure(methods_all, record_all, prob, mypath):
    """
    Plots generator.
    Input: 
        methods_all: a list contains all methods
        record_all: a list contains all record matrix of listed methods, 
        s.t., [fx, norm(gx), oracle calls, time, stepsize, is_negative_curvature]
        prob: name of problem
        mypath: directory path for saving plots
    OUTPUT:
        Oracle calls vs. F
        Oracle calls vs. Gradient norm
        Iteration vs. Step Size
    """
    fsize = 24
    myplt2 = plt.loglog
    myplt = plt.semilogy
    
    figsz = (10,6)
    mydpi = 300
    
    fig1 = plt.figure(figsize=figsz)
    
    F_star = min(record_all[0][-1,0], record_all[1][-1,0],
                  record_all[2][-1,0], record_all[3][-1,0], record_all[4][-1,0],
                  record_all[5][-1,0], record_all[6][-1,0])
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        record = record[:sum(record[:,2] < 3000),:]
        record[:,0] = (record[:,0] - F_star)/max(F_star, 1)
        if methods_all[i] == 'GD' or methods_all[i] == 'AndersonAcc_pure':
            record = record[:,:5]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0], record[:,2]+1)
    plt.xlabel('Oracle calls', fontsize=fsize)
    plt.ylabel(r'$\frac{f(x_k) - f^{*}}{\max \{f^{*}, 1\}}$', fontsize=fsize)
    plt.legend()
    fig1.savefig(os.path.join(mypath, 'F'), dpi=mydpi)
    
    
    fig2 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        record = record[:sum(record[:,2] < 3000),:]
        if methods_all[i] == 'GD' or methods_all[i] == 'AndersonAcc_pure':
            record = record[:,:5]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,1], record[:,2]+1)
    plt.xlabel('Oracle calls', fontsize=fsize)
    plt.ylabel(r'$|| \nabla f(x_k) ||$', fontsize=fsize)
    plt.legend()
    fig2.savefig(os.path.join(mypath, 'Gradient_norm'), dpi=mydpi)
    
    fig3 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        record[:,0] = (record[:,0] - F_star)/max(F_star, 1)
        record = record[:sum(record[:,3] < 3),:]
        # print(record[:,2]<1E-7))
        if np.where(record[:,1]<1E-7)[0] != []:
            record = record[:min(np.where(record[:,1]<1E-7)[0][0]+1, len(record[:,2])),:]
        if methods_all[i] == 'GD' or methods_all[i] == 'AndersonAcc_pure':
            record = record[:,:5]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0], record[:,3])
    plt.xlabel('Time', fontsize=fsize)
    plt.ylabel(r'$\frac{f(x_k) - f^{*}}{\max \{f^{*}, 1\}}$', fontsize=fsize)
    plt.legend()
    fig3.savefig(os.path.join(mypath, 'Time'), dpi=mydpi)