import torch

from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh, artanh, tanh
from manifolds.hyperboloid import Hyperboloid
from manifolds.euclidean import Euclidean
from manifolds.poincare import PoincareBall


class Mixture(Manifold):
    """
    Mixture manifold class.

    """

    def __init__(self):
        super(Mixture, self).__init__()
        self.name = 'Mixture'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6
        self.Hyperboloid = Hyperboloid()
        self.Euclidean = Euclidean()
        self.Poincare = PoincareBall()
        self.Fractions = [1/3,1/3,1/3]
        self.length = 0
        self.Split = [0,0,0]

    def sqdist(self, x, y, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.sqdist(x, y, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.sqdist(x, y, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.sqdist(x, y, c)
        self.rescale_dims(x)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.sqdist(x[..., :self.Split[0]], y[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.sqdist(x[..., self.Split[0] : self.Split[1]], y[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.sqdist(x[..., self.Split[1] : self.Split[2]], x[..., self.Split[1] : self.Split[2]], c)
        ##sum lol
        sum = 0
        if self.Fractions[0] != 0: sum += hyper.view(-1)
        if self.Fractions[1] != 0: sum += euc
        if self.Fractions[2] != 0: sum += poin
        return sum

    def proj(self, x, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.proj(x, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.proj(x, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.proj(x, c)
        self.rescale_dims(x)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.proj(x[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.proj(x[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.proj(x[..., self.Split[1] : self.Split[2]], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)

    def proj_tan(self, u, x, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.proj_tan(u, x, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.proj_tan(u, x, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.proj_tan(u, x, c)
        self.rescale_dims(x)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.proj_tan(u[..., :self.Split[0]], x[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.proj_tan(u[..., self.Split[0] : self.Split[1]], x[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.proj_tan(u[..., self.Split[1] : self.Split[2]], x[..., self.Split[1] : self.Split[2]], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)

    def proj_tan0(self, u, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.proj_tan0(u, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.proj_tan0(u, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.proj_tan0(u, c)
        self.rescale_dims(u)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.proj_tan0(u[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.proj_tan0(u[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.proj_tan0(u[..., self.Split[1] : self.Split[2]], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)

    def expmap(self, u, x, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.expmap(u, x, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.expmap(u, x, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.expmap(u, x, c)
        self.rescale_dims(x)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.expmap(u[..., :self.Split[0]], x[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.expmap(u[..., self.Split[0] : self.Split[1]], x[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.expmap(u[..., self.Split[1] : self.Split[2]], x[..., self.Split[1] : self.Split[2]], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)


    def logmap(self, x, y, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.logmap(x, y, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.logmap(x, y, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.logmap(x, y, c)
        self.rescale_dims(x)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.logmap(x[..., :self.Split[0]], y[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.logmap(x[..., self.Split[0] : self.Split[1]], y[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.logmap(x[..., self.Split[1] : self.Split[2]], y[..., self.Split[1] : self.Split[2]], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)

    def expmap0(self, u, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.expmap0(u, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.expmap0(u, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.expmap0(u, c)
        self.rescale_dims(u)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.expmap0(u[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.expmap0(u[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.expmap0(u[..., self.Split[1] : self.Split[2]], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)

    def logmap0(self, x, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.logmap0(x, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.logmap0(x, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.logmap0(x, c)
        self.rescale_dims(x)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.logmap0(x[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.logmap0(x[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.logmap0(x[..., self.Split[1] : self.Split[2]], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)

    def mobius_add(self, x, y, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.mobius_add(x, y, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.mobius_add(x, y, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.mobius_add(x, y, c)
        self.rescale_dims(x)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.mobius_add(x[..., :self.Split[0]], y[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.mobius_add(x[..., self.Split[0] : self.Split[1]], y[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.mobius_add(x[..., self.Split[1] : self.Split[2]], y[..., self.Split[1] : self.Split[2]], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)

    def mobius_matvec(self, m, x, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.mobius_matvec(m, x, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.mobius_matvec(m, x, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.mobius_matvec(m, x, c)

        self.rescale_dims(x)
        'Seperate out Manifolds, project:'

        'hyperboloid'
        if self.Fractions[0] != 0:
            hyper_u = self.Hyperboloid.logmap0(x[..., :self.Split[0]], c)
            'accumulate'
            accumulated_u = torch.cat([hyper_u, x[..., self.Split[0] : self.Split[2]]], dim=1)
        else:
            accumulated_u = x

        'multiply'
        mu = accumulated_u @ m.transpose(-1, -2)
        self.rescale_dims(mu)


        'now unscrew the vector'

        'hyperboloid'
        if self.Fractions[0] != 0:
            hyper_final = self.Hyperboloid.expmap0(mu[..., :self.Split[0]], c)

        if self.Fractions[2] != 0:
            'poincare'
            sqrt_c = c ** 0.5
            poinc_x_norm = x[..., self.Split[1] : self.Split[2]].norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
            poinc_mx_norm = mu[..., self.Split[1] : self.Split[2]].norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
            res_c = tanh(poinc_x_norm / poinc_x_norm * artanh(sqrt_c * poinc_x_norm)) * mu[..., self.Split[1] : self.Split[2]] / (poinc_x_norm * sqrt_c)
            cond = (mu[..., self.Split[1] : self.Split[2]] == 0).prod(-1, keepdim=True, dtype=torch.uint8)
            res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
            poincare_final = torch.where(cond, res_0, res_c)

        if self.Fractions[0] == 0:
            new_mu = torch.cat([mu[...,self.Split[0] : self.Split[1]], poincare_final], dim=1)
        elif self.Fractions[1] == 0:
            new_mu = torch.cat([hyper_final, poincare_final], dim=1)
        elif self.Fractions[2] == 0:
            new_mu = torch.cat([hyper_final, mu[...,self.Split[0] : self.Split[1]]], dim=1)
        else:
            new_mu = torch.cat([hyper_final, mu[...,self.Split[0] : self.Split[1]], poincare_final], dim=1)

        return new_mu

        # mu = u @ m.transpose(-1, -2)
        #
        #
        #
        #
        # hyper = self.Hyperboloid.mobius_matvec(m[ :self.Split[0], :], x[..., :self.Split[0]], c)
        # euc = self.Euclidean.mobius_matvec(m[ self.Split[0] : self.Split[1], :], x[..., self.Split[0] : self.Split[1]], c)
        # poin = self.Poincare.mobius_matvec(m[ self.Split[1] : self.Split[2], :], x[..., self.Split[1] : self.Split[2]], c)
        #
        # # print(hyper.shape)
        # # print(euc.shape)
        # # print(poin.shape)
        # return torch.cat([hyper, euc, poin], dim = 1)

    def rescale_dims(self, x):
        length = len(x[0,:])
        if length != self.length:
            self.Split[0] = int(self.Fractions[0] * length)
            if self.Fractions[2] == 0:
                self.Split[1] = length
            else:
                self.Split[1] = int(self.Fractions[1] * length) + self.Split[0]
            self.Split[2] = length
            self.length = length


    def ptransp(self, x, y, u, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.ptransp(x, y, u, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.ptransp(x, y, u, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.ptransp(x, y, u, c)
        self.rescale_dims(x)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.ptransp(x[..., :self.Split[0]], y[..., :self.Split[0]], u[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.ptransp(x[..., self.Split[0] : self.Split[1]], y[..., self.Split[0] : self.Split[1]], u[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.ptransp(x[..., self.Split[1] : self.Split[2]], y[..., self.Split[1] : self.Split[2]], u[..., self.Split[1] : self.Split[2]], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)

    def ptransp0(self, x, u, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.ptransp0(x, u, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.ptransp0(x, u, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.ptransp0(x, u, c)
        self.rescale_dims(x)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.ptransp0(x[..., :self.Split[0]], u[..., :self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.ptransp0(x[..., self.Split[0] : self.Split[1]], u[..., self.Split[0] : self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.ptransp0(x[..., self.Split[1] : self.Split[2]], u[..., self.Split[1] : self.Split[2]], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)


    def normalize(self, p):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return p
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.normalize(p)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return p
        self.rescale_dims(p)
        if self.Fractions[0] != 0:
            hyper = p[...,:self.Split[0]]
        if self.Fractions[1] != 0:
            euc = self.Euclidean.normalize(p[...,self.Split[0]:self.Split[1]])
        if self.Fractions[2] != 0:
            poin = p[...,self.Split[1]:self.Split[2]]
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)

    
    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w
    # def to_poincare(self, x, c):
    #     K = 1. / c
    #     sqrtK = K ** 0.5
    #     d = x.size(-1) - 1
    #     return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)
