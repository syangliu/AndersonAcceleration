# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:47:45 2021

@author: Liu Yang
"""

import torch

class Anderson:
    def __init__(self, x0, num):
        dType = x0.dtype
        device = x0.device
        self.mk = num
        self.dim = len(x0)
        self.current_F_ = x0.clone()
        self.prev_dG_ = torch.zeros(num, self.dim, dtype=dType, device=device)
        self.prev_dF_ = torch.zeros(num, self.dim, dtype=dType, device=device)
        self.M_ = torch.zeros(num, num, dtype=dType, device=device)  # num*num array
        self.theta_ = torch.zeros(num, dtype=dType, device=device)
        self.dF_scale_ = torch.zeros(num, dtype=dType, device=device)
        self.dG_scale_ = torch.zeros(num, dtype=dType, device=device)
        self.current_u_ = x0.clone()
        self.iter_ = 0
        self.cond_num = 0
        self.col_idx_ = 0

    def compute(self, g, eta=0, cond=False):
        G = g.clone()
        self.current_F_ = G - self.current_u_
#        print(self.current_u_)

        if self.iter_ == 0:
            self.prev_dF_[0, :] = -self.current_F_
            self.prev_dG_[0, :] = - G
            self.current_u_ = G.clone()
        else:
            self.prev_dF_[self.col_idx_, :] += self.current_F_.clone()
            self.prev_dG_[self.col_idx_, :] += G

            eps = 1e-14
            norm = self.prev_dF_[self.col_idx_, :].norm()
#            print(self.prev_dF_)
            scale = max(norm, eps)
            self.dF_scale_[self.col_idx_] = scale
            self.prev_dF_[self.col_idx_, :] /= scale

            m_k = min(self.iter_, self.mk)

            if m_k == 1:
                self.theta_[0] = 0
                dF_norm = self.prev_dF_[self.col_idx_, :].norm()
                self.M_[0, 0] = dF_norm * dF_norm
                coef = self.M_[0, 0] + eta * (dF_norm ** 2 + (self.prev_dF_[self.col_idx_,:]-self.prev_dG_[self.col_idx_,:]/scale).norm()**2)
                self.cond_num = 1
                if dF_norm > eps:
                    self.theta_[0] = torch.dot(self.prev_dF_[self.col_idx_, :] / dF_norm, self.current_F_[:] / dF_norm)
            else:
                new_inner_prod = torch.mv(self.prev_dF_[0:m_k, :], self.prev_dF_[self.col_idx_, :])
                self.M_[self.col_idx_, 0:m_k] = new_inner_prod
                self.M_[0:m_k, self.col_idx_] = new_inner_prod

                b = torch.mv(self.prev_dF_[0:m_k, :], self.current_F_)
                if cond:
                    eigenvalue = torch.eig(self.M_[0:m_k, 0:m_k].T @ self.M_[0:m_k, 0:m_k])[0][:,0]
                    self.cond_num = torch.sqrt(max(abs(eigenvalue))/min(abs(eigenvalue)))
                self.theta_[0:m_k] = torch.pinverse(self.M_[0:m_k, 0:m_k] + eta * (
                        torch.norm(self.prev_dF_[0:m_k,:],'fro')**2+torch.norm(
                                self.prev_dF_[0:m_k,:]-self.prev_dG_[0:m_k,:]/self.dF_scale_[
                                        0:m_k,None],'fro')**2 )*torch.eye(m_k, device=g.device, dtype=torch.float64)) @ b

            v = self.theta_[0:m_k] / self.dF_scale_[0:m_k]
#            print(self.dF_scale_[0:m_k], v)
            self.current_u_ = G - torch.mv(self.prev_dG_[0:m_k, :].T, v)

            self.col_idx_ = (self.col_idx_ + 1) % self.mk
            self.prev_dF_[self.col_idx_, :] = -self.current_F_
            self.prev_dG_[self.col_idx_, :] = -G
        self.iter_ += 1

        return self.current_u_.clone()

    def replace(self, x):
        self.current_u_ = x.clone()

    def reset(self, x):
        self.current_u_ = x.clone()
        self.iter_ = 0
        self.cond_num = 0
        self.col_idx_ = 0
        
        
def main():
    torch.manual_seed(1)
    d = 20
    W = torch.randn(d, d, dtype=torch.float64)
    A = W.T @ W
    b = torch.randn(d, dtype=torch.float64)
    f = lambda x: -(x @ b) + (x @ (A @ x))/2
    g = lambda x: (A @ x) - b
    print(f(torch.inverse(A)@b))
    x = torch.zeros(d, dtype=torch.float64)+2
    L = A.norm()
    m = 5
    maxIter = 60
    acc=Anderson(x,m)
#    iters = 0
#    while iters <= maxIter:
#        gx = x - g(x)/L
#        xn = acc.compute(gx)
#        iters += 1
#        if f(xn) < f(x):
#            x = xn
#        else:
#            x = gx
#        print('f', f(x))
    iters2 = 0
    acc=Anderson(x,2*m)
    while iters2 <= maxIter:
        gx = x - g(x)/L
        xn = acc.compute(gx,1e-8)
        if f(xn) < f(x):
            x = xn
            print('acc')
        else:
            x = gx
            print('rej')
        if iters2 % (2*m) == 0:
            acc.reset(x)
        print('f', f(x))
        iters2 = iters2 + 1
        
if __name__ == '__main__':
    main()
    