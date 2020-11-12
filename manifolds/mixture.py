import torch

from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh
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
        self.length = None
        self.Fractions = [1/3,1/3,1/3]

        self.Split = [45,90,128]

    def sqdist(self, x, y, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.sqdist(x[..., :self.Split[0]], y[..., :self.Split[0]], c)
        euc = self.Euclidean.sqdist(x[..., self.Split[0] : self.Split[1]], y[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.sqdist(x[..., self.Split[1] : self.Split[2]], x[..., self.Split[1] : self.Split[2]], c)
        total = (hyper**2 + euc**2 + poin**2)**.5
        ##sum lol
        return hyper.view(-1) + euc + poin

    def proj(self, x, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.proj(x[..., :self.Split[0]], c)
        euc = self.Euclidean.proj(x[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.proj(x[..., self.Split[1] : self.Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def proj_tan(self, u, x, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.proj_tan(u[..., :self.Split[0]], x[..., :self.Split[0]], c)
        euc = self.Euclidean.proj_tan(u[..., self.Split[0] : self.Split[1]], x[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.proj_tan(u[..., self.Split[1] : self.Split[2]], x[..., self.Split[1] : self.Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def proj_tan0(self, u, c):
        self.rescale_dims(u)
        hyper = self.Hyperboloid.proj_tan0(u[..., :self.Split[0]], c)
        euc = self.Euclidean.proj_tan0(u[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.proj_tan0(u[..., self.Split[1] : self.Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def expmap(self, u, x, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.expmap(u[..., :self.Split[0]], x[..., :self.Split[0]], c)
        euc = self.Euclidean.expmap(u[..., self.Split[0] : self.Split[1]], x[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.expmap(u[..., self.Split[1] : self.Split[2]], x[..., self.Split[1] : self.Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)


    def logmap(self, x, y, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.logmap(x[..., :self.Split[0]], y[..., :self.Split[0]], c)
        euc = self.Euclidean.logmap(x[..., self.Split[0] : self.Split[1]], y[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.logmap(x[..., self.Split[1] : self.Split[2]], y[..., self.Split[1] : self.Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def expmap0(self, u, c):
        self.rescale_dims(u)
        hyper = self.Hyperboloid.expmap0(u[..., :self.Split[0]], c)
        euc = self.Euclidean.expmap0(u[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.expmap0(u[..., self.Split[1] : self.Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def logmap0(self, x, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.logmap0(x[..., :self.Split[0]], c)
        euc = self.Euclidean.logmap0(x[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.logmap0(x[..., self.Split[1] : self.Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def mobius_add(self, x, y, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.mobius_add(x[..., :self.Split[0]], y[..., :self.Split[0]], c)
        euc = self.Euclidean.mobius_add(x[..., self.Split[0] : self.Split[1]], y[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.mobius_add(x[..., self.Split[1] : self.Split[2]], y[..., self.Split[1] : self.Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def mobius_matvec(self, m, x, c):
        self.rescale_dims(x)
        'Seperate out Manifolds, project:'

        'hyperboloid'
        hyper_u = self.Hyperboloid.logmap0(x[..., :self.Split[0]], c)

        'accumulate'
        accumulated_u = torch.cat([hyper_u, x[..., self.Split[0] : self.Split[2]]], dim=1)

        'multiply'
        mu = accumulated_u @ m.transpose(-1, -2)
        self.rescale_dims(mu)


        'now unscrew the vector'

        'hyperboloid'
        mu[..., self.Split[0]] = self.Hyperboloid.expmap0(mu[..., self.Split[0]], c)


        'poincare'
        sqrt_c = c ** 0.5
        poinc_x_norm = x[..., self.Split[1] : self.Split[2]].norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)

        poinc_mx_norm = mu[..., self.Split[1] : self.Split[2]].norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(poinc_x_norm / poinc_x_norm * artanh(sqrt_c * poinc_x_norm)) * mu[..., self.Split[1] : self.Split[2]] / (poinc_x_norm * sqrt_c)
        cond = (mu[..., self.Split[1] : self.Split[2]] == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        mu[..., self.Split[1] : self.Split[2]] = torch.where(cond, res_0, res_c)

        return mu

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
            self.Split[1] = int(self.Fractions[1] * length) + self.Split[0]
            self.Split[2] = length
            self.length = length







    def ptransp(self, x, y, u, c):
        hyper = self.Hyperboloid.ptransp(x[..., :self.Split[0]], y[..., :self.Split[0]], u[..., :self.Split[0]], c)
        euc = self.Euclidean.ptransp(x[..., self.Split[0] : self.Split[1]], y[..., self.Split[0] : self.Split[1]], u[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.ptransp(x[..., self.Split[1] : self.Split[2]], y[..., self.Split[1] : self.Split[2]], u[..., self.Split[1] : self.Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def ptransp0(self, x, u, c):
        hyper = self.Hyperboloid.ptransp0(x[..., :self.Split[0]], u[..., :self.Split[0]], c)
        euc = self.Euclidean.ptransp0(x[..., self.Split[0] : self.Split[1]], u[..., self.Split[0] : self.Split[1]], c)
        poin = self.Poincare.ptransp0(x[..., self.Split[1] : self.Split[2]], u[..., self.Split[1] : self.Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)


    def normalize(self, p):
        return torch.cat([p[...,:self.Split[0]], self.Euclidean.normalize(p[...,self.Split[0]:self.Split[1]]), p[...,self.Split[1]:self.Split[2]]], dim=1)

    # def to_poincare(self, x, c):
    #     K = 1. / c
    #     sqrtK = K ** 0.5
    #     d = x.size(-1) - 1
    #     return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)
