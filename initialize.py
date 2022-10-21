import numpy as np
import torch
from optim_algo import AndersonAcc, L_BFGS
from showFigure import showFigure
from sklearn import preprocessing
from student_t import student_t
from regularizer import regConvex, regNonconvex
import os
from least_square import least_square
from scipy import sparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
        
def initialize(data, methods, prob, x0Type, algPara, lamda=1): 
    """
    data: name of chosen dataset
    methods: name of chosen algorithms
    prob: name of chosen objective problems
    regType: type of regularization 
    x0Type: type of starting point
    algPara: a class that contains:
        mainLoopMaxItrs: maximum iterations for main loop
        funcEvalMax: maximum oracle calls (function evaluations) for algorithms
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        show: print result for every iteration
    lamda: parameter of regularizer
    """
    print('Initialization...')
    prob = prob[0]
    x0Type = x0Type[0]
        
    print('Problem:', prob, end='  ')
    if hasattr(algPara, 'cutData'):
        print('Data-size using: ', algPara.cutData)
    print('regulization = %8s' % algPara.regType, end='  ')
    print('gradTol = %8s' % algPara.gradTol, end='  ')
    print('Starting point = %8s ' % x0Type)  
    algPara.regType = algPara.regType[0]
    if algPara.regType == 'None':
        reg = None
    if algPara.regType == 'Convex':
        reg = lambda x: regConvex(x, lamda)
    if algPara.regType == 'Nonconvex':
        reg = lambda x: regNonconvex(x, lamda)
    if algPara.regType == 'Nonsmooth':
        reg = None
          
    filename = '%s_%s_reg_%s_Orc_%s_x0_%s_reg_%s_ZOOM_%s_m_%s_L_%s_Nu_%s' % (
            prob, data, algPara.regType, algPara.funcEvalMax, x0Type, lamda, 
            algPara.zoom, algPara.Andersonm, algPara.L, algPara.student_t_nu) 
        
    mypath = filename
    print('filename', filename)
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    
    execute(data, methods, x0Type, algPara, reg, mypath, prob, lamda)

def execute(data, methods, x0Type, algPara, reg, mypath, prob, lamda):  
    """
    Excute all methods/problems with 1 total run and give plots.
    """            
    
    data_dir = '../Data'  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print('device', device)
    print('Dataset:', data[0])
    if data[0] == 'cifar10':
        train_Set = datasets.CIFAR10(data_dir, train=True,
                                    transform=transforms.ToTensor(), 
                                    download=True)  
        (n, r, c, rgb) = train_Set.data.shape
        d = rgb*r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = torch.DoubleTensor(X)
        Y_index = train_Set.targets  
        Y_index = torch.DoubleTensor(Y_index)
    if data[0] == 'mnist':
        train_Set = datasets.MNIST(data_dir, train=True,
                                    transform=transforms.ToTensor(), 
                                    download=True)
        (n, r, c) = train_Set.data.shape
        d = r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = X.double()
        Y_index = train_Set.targets  
        Y_index = Y_index.double()
        
    if data[0] == 'CelebA':
        train_Set = datasets.CelebA(data_dir, split ='train', target_type='attr',
                                    transform=transforms.ToTensor(), 
                                    download=True)
        (n, r, c) = train_Set.data.shape
        d = r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = X.double()
        Y_index = train_Set.targets  
        Y_index = Y_index.double()
        
    if data[0] == 'stl10':
        train_Set = datasets.STL10(data_dir,split='train',
                                   transform=transforms.ToTensor(), 
                                   download=True)
        (n, r, c, rgb) = train_Set.data.shape
        d = rgb*r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = torch.DoubleTensor(X)
        Y_index = train_Set.labels
        Y_index = torch.DoubleTensor(Y_index)
        
    if data[0] == 'caltech256':
        train_Set = datasets.Caltech256(data_dir,
                                    transform=transforms.ToTensor(), 
                                    download=True)
        (n, r, c) = train_Set.data.shape
        d = r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = X.double()
        Y_index = train_Set.targets  
        Y_index = Y_index.double()
        
    if data[0] == 'caltech101':
        train_Set = datasets.Caltech101(data_dir,
                                    transform=transforms.ToTensor(), 
                                    download=True)
        (n, r, c) = train_Set.data.shape
        d = r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = X.double()
        Y_index = train_Set.targets  
        Y_index = Y_index.double()
        
    if data[0] == 'fmnist':
        train_Set = datasets.FashionMNIST(data_dir, train=True,
                                    transform=transforms.ToTensor(), 
                                    download=True)
        (n, r, c) = train_Set.data.shape
        d = r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = X.double()
        Y_index = train_Set.targets  
        Y_index = Y_index.double()
    
    if prob == 'nls' or 'student_t':
        Y = (Y_index%2!=0).double()*1
        l = d
    if prob == 'softmax':
        I = torch.eye(total_C, total_C - 1)
#        Y = (Y_index%2==0).double()*1
        Y = I[np.array(Y_index), :]
        Y = Y.double()
        l = d*(total_C - 1)
        
    spnorm = np.linalg.norm(X, 2)
    if prob == 'nls':
        algPara.L_g=spnorm**2/4/X.shape[0] + lamda
    if prob == 'student_t':
        algPara.L_g=2*spnorm**2/X.shape[0]/algPara.student_t_nu + lamda
        
    X = X.to(device)
    Y = Y.to(device)
    
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    print('Lipschiz', algPara.L_g)
    print('Original_Dataset_shape:', X.shape, end='  ') 
            
    if prob == 'softmax':      
        obj = lambda x, control=None, HProp=1: softmax(
                X, Y, x, HProp, control, reg)  
        
    if prob == 'nls':
        obj = lambda x, control=None, HProp=1: least_square(
                X, Y, x, HProp, control, reg)
        
    if prob == 'student_t':
        obj = lambda x, control=None, HProp=1: student_t(
                X, Y, x, nu=algPara.student_t_nu, HProp=HProp, 
                arg=control, reg=reg)
        
    if prob == 'logitreg':
        obj = lambda x, control=None, HProp=1: logitreg(
                X, Y, x, HProp, control, reg)
    x0 = generate_x0(x0Type, l, zoom=algPara.zoom, dType=algPara.dType)  
    x0 = x0.to(device)
    
    methods_all, record_all = run_algorithms(
            obj, x0, methods, algPara, mypath)
    showFigure(methods_all, record_all, prob, mypath)
        
def run_algorithms(obj, x0, methods, algPara, mypath):
    """
    Distribute all problems to its cooresponding optimisation methods.
    """
    record_all = []            
    record_txt = lambda filename, myrecord: np.savetxt(
            os.path.join(mypath, filename+'.txt'), myrecord.cpu(), delimiter=',') 
        
    if 'AndersonAcc' in methods:
        print(' ')
        myMethod = 'AndersonAcc'
        arg = ['general', 'pure', 'residual', 'GD']
        flag = 1
        i = 0
        maxOC = algPara.funcEvalMax
        while flag == 1 and i < (len(arg)):
            arg_i = arg[i]            
            if i == 3:
                myMethod = 'GD'
            else:
                myMethod = 'AndersonAcc_%s_restart' % (arg_i)
            print(myMethod)
            x, record = AndersonAcc(
                    obj, x0, algPara.Andersonm, algPara.L_g, algPara.mainLoopMaxItrs, 
                    maxOC, algPara.gradTol, algPara.show, arg_i, record_txt)
            if algPara.savetxt is True:
                np.savetxt(os.path.join(mypath, myMethod+'.txt'), record.cpu(), delimiter=',')
            record_all.append(myMethod)
            record_all.append(record.cpu())
            
            if i == 1 or i == 2:    
                myMethod = 'AndersonAcc_%s' % (arg_i)
                print(myMethod)
                x, record = AndersonAcc(
                        obj, x0, algPara.Andersonm, algPara.L_g, algPara.mainLoopMaxItrs, 
                        maxOC, algPara.gradTol, algPara.show, arg_i, record_txt, False)
                if algPara.savetxt is True:
                    np.savetxt(os.path.join(mypath, myMethod+'.txt'), record.cpu(), delimiter=',')
                record_all.append(myMethod)
                record_all.append(record.cpu())
            
            i += 1
    
    if 'L_BFGS' in methods and myMethod != 'abort':
        print(' ')
        myMethod = 'L_BFGS'
        print(myMethod)
        record_all.append(myMethod)
        x, record = L_BFGS(
                obj, x0, algPara.mainLoopMaxItrs, maxOC, 
                algPara.lineSearchMaxItrs, algPara.gradTol, algPara.L, algPara.beta, 
                algPara.beta2, algPara.show, record_txt)
        if algPara.savetxt is True:
            np.savetxt(os.path.join(mypath, 'L_BFGS.txt'), record.cpu(), delimiter=',')
        record_all.append(record.cpu())
            
    methods_all = record_all[::2]
    record_all = record_all[1::2]
    
    return methods_all, record_all

        
def sofmax_init(train_X, train_Y):
    """
    Initialize data matrix for softmax problems.
    For multi classes classification.
    INPUT:
        train_X: raw training data
        train_Y: raw label data
    OUTPUT:
        train_X: DATA matrix
        Y: label matrix
        l: dimensions
    """
    n, d= train_X.shape
    Classes = sorted(set(train_Y))
    Total_C  = len(Classes)
    if Total_C == 2:
        train_Y = (train_Y == 1)*1
    l = d*(Total_C-1)
    I = np.ones(n)
    
    X_label = np.array([i for i in range(n)])
    Y = sparse.coo_matrix((I,(X_label, train_Y)), shape=(
            n, Total_C)).tocsr().toarray()
    Y = Y[:,:-1]
    return train_X, Y, l    

        
def nls_init(train_X, train_Y, idx=5):
    """
    Initialize data matrix for non-linear least square problems.
    For binary classification.
    INPUT:
        train_X: raw training data
        train_Y: raw label data
        idx: a number s.t., relabelling index >= idx classes into 1, the rest 0. 
    OUTPUT:
        train_X: DATA matrix
        Y: label matrix
        l: dimensions
    """
    n, d= train_X.shape
    Y = (train_Y >= idx)*1 #bool to int
    Y = Y.reshape(n,1)
    l = d
    return train_X, Y, l

def scale_train_X(train_X, standarlize=False, normalize=False): 
    """
    Standarlization/Normalization of trainning DATA.
    """
    if standarlize:
        train_X = preprocessing.scale(train_X)            
    if normalize:
        train_X = preprocessing.normalize(train_X, norm='l2')
    return train_X

    
def generate_x0(x0Type, l, zoom=1, dType=torch.double, dvc = 'cpu'):    
    """
    Generate different type starting point.
    """
    if x0Type == 'randn':
        x0 = torch.randn(l, dtype=dType, device=dvc)/zoom
    if x0Type == 'rand':
        x0 = torch.rand(l, dtype=dType, device=dvc)/zoom
    if x0Type == 'ones':
        x0 = torch.ones(l, dtype=dType, device=dvc)
    if x0Type == 'zeros':
        x0 = torch.zeros(l, dtype=dType, device=dvc)
    return x0