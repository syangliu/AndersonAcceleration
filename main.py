from initialize import initialize
import torch

class algPara():
    def __init__(self, value):
        self.value = value

def main(data, prob, lamda):
    #initialize methods
    
    methods = [
            'AndersonAcc',
              'L_BFGS',
            ]
    
    algPara.regType = [
    #        'None',
            'Convex',
    #        'Nonconvex',
            ] 
    
    #initial point
    x0Type = [
            'randn',
    #        'rand',
    #        'ones',
    #        'zeros',
            ]
    #initialize parameter
    algPara.funcEvalMax = 2E3 #Set mainloop stops with Maximum Function Evaluations
    algPara.mainLoopMaxItrs = 1E5 #Set mainloop stops with Maximum Iterations
    algPara.gradTol = 1e-7 #If norm(g)<gradTol, minFunc loop breaks
    algPara.lineSearchMaxItrs = 1E3
    algPara.L = 20
    algPara.beta = 1E-4
    algPara.beta2 = 0.9
    algPara.zoom = 1
    algPara.student_t_nu = 20
    # algPara.cutData = 100
    algPara.dType = torch.float64
    algPara.savetxt = True
    algPara.show = True
    algPara.Andersonm = algPara.L
    
    ## Initialize
    initialize(data, methods, prob, x0Type, algPara, lamda)
        
    
    
if __name__ == '__main__':
    # dataset, problem, lambda for regularizer
    main(['cifar10'], ['nls'], 1E-2)
    # main(['cifar10'], ['student_t'], 1E-2)
    # main(['stl10'], ['nls'], 1E-1)
    # main(['stl10'], ['student_t'], 1E-1)