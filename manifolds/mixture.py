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

    def sqdist(self, x, y, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.sqdist(x[..., :Split[0]], y[..., :Split[0]], c)
        euc = self.Euclidean.sqdist(x[..., Split[0] : Split[1]], y[..., Split[0] : Split[1]], c)
        poin = self.Poincare.sqdist(x[..., Split[1] : Split[2]], x[..., Split[1] : Split[2]], c)
        total = (hyper**2 + euc**2 + poin**2)**.5
        ##sum lol
        return hyper.view(-1) + euc + poin

    def proj(self, x, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.proj(x[..., :Split[0]], c)
        euc = self.Euclidean.proj(x[..., Split[0] : Split[1]], c)
        poin = self.Poincare.proj(x[..., Split[1] : Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def proj_tan(self, u, x, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.proj_tan(u[..., :Split[0]], x[..., :Split[0]], c)
        euc = self.Euclidean.proj_tan(u[..., Split[0] : Split[1]], x[..., Split[0] : Split[1]], c)
        poin = self.Poincare.proj_tan(u[..., Split[1] : Split[2]], x[..., Split[1] : Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def proj_tan0(self, u, c):
        self.rescale_dims(u)
        hyper = self.Hyperboloid.proj_tan0(u[..., :Split[0]], c)
        euc = self.Euclidean.proj_tan0(u[..., Split[0] : Split[1]], c)
        poin = self.Poincare.proj_tan0(u[..., Split[1] : Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def expmap(self, u, x, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.expmap(u[..., :Split[0]], x[..., :Split[0]], c)
        euc = self.Euclidean.expmap(u[..., Split[0] : Split[1]], x[..., Split[0] : Split[1]], c)
        poin = self.Poincare.expmap(u[..., Split[1] : Split[2]], x[..., Split[1] : Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)


    def logmap(self, x, y, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.logmap(x[..., :Split[0]], y[..., :Split[0]], c)
        euc = self.Euclidean.logmap(x[..., Split[0] : Split[1]], y[..., Split[0] : Split[1]], c)
        poin = self.Poincare.logmap(x[..., Split[1] : Split[2]], y[..., Split[1] : Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def expmap0(self, u, c):
        self.rescale_dims(u)
        hyper = self.Hyperboloid.expmap0(u[..., :Split[0]], c)
        euc = self.Euclidean.expmap0(u[..., Split[0] : Split[1]], c)
        poin = self.Poincare.expmap0(u[..., Split[1] : Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def logmap0(self, x, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.logmap0(x[..., :Split[0]], c)
        euc = self.Euclidean.logmap0(x[..., Split[0] : Split[1]], c)
        poin = self.Poincare.logmap0(x[..., Split[1] : Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def mobius_add(self, x, y, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.mobius_add(x[..., :Split[0]], y[..., :Split[0]], c)
        euc = self.Euclidean.mobius_add(x[..., Split[0] : Split[1]], y[..., Split[0] : Split[1]], c)
        poin = self.Poincare.mobius_add(x[..., Split[1] : Split[2]], y[..., Split[1] : Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def mobius_matvec(self, m, x, c):
        self.rescale_dims(x)
        'Seperate out Manifolds, project:'

        'hyperboloid'
        hyper_u = self.Hyperboloid.logmap0(x[..., :Split[0]], c)

        'accumulate'
        accumulated_u = torch.cat([hyper_u, x[..., Split[0] : Split[2]]], dim=1)

        'multiply'
        mu = accumulated_u @ m.transpose(-1, -2)
        self.rescale_dims(mu)


        'now unscrew the vector'

        'hyperboloid'
        # print(mu[..., :Split[0]].shape)
        hyper_final = self.Hyperboloid.expmap0(mu[..., :Split[0]], c)


        'poincare'
        sqrt_c = c ** 0.5
        poinc_x_norm = x[..., Split[1] : Split[2]].norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)

        poinc_mx_norm = mu[..., Split[1] : Split[2]].norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(poinc_x_norm / poinc_x_norm * artanh(sqrt_c * poinc_x_norm)) * mu[..., Split[1] : Split[2]] / (poinc_x_norm * sqrt_c)
        cond = (mu[..., Split[1] : Split[2]] == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        poincare_final = torch.where(cond, res_0, res_c)

        new_mu = torch.cat([hyper_final, mu[...,Split[0] : Split[1]], poincare_final], dim=1)

        return new_mu

        # mu = u @ m.transpose(-1, -2)
        #
        #
        #
        #
        # hyper = self.Hyperboloid.mobius_matvec(m[ :Split[0], :], x[..., :Split[0]], c)
        # euc = self.Euclidean.mobius_matvec(m[ Split[0] : Split[1], :], x[..., Split[0] : Split[1]], c)
        # poin = self.Poincare.mobius_matvec(m[ Split[1] : Split[2], :], x[..., Split[1] : Split[2]], c)
        #
        # # print(hyper.shape)
        # # print(euc.shape)
        # # print(poin.shape)
        # return torch.cat([hyper, euc, poin], dim = 1)

    def rescale_dims(self, x):
        length = len(x[0,:])
        if length != self.length:
            Split[0] = int(self.Fractions[0] * length)
            Split[1] = int(self.Fractions[1] * length) + Split[0]
            Split[2] = length
            self.length = length


    def ptransp(self, x, y, u, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.ptransp(x[..., :Split[0]], y[..., :Split[0]], u[..., :Split[0]], c)
        euc = self.Euclidean.ptransp(x[..., Split[0] : Split[1]], y[..., Split[0] : Split[1]], u[..., Split[0] : Split[1]], c)
        poin = self.Poincare.ptransp(x[..., Split[1] : Split[2]], y[..., Split[1] : Split[2]], u[..., Split[1] : Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)

    def ptransp0(self, x, u, c):
        self.rescale_dims(x)
        hyper = self.Hyperboloid.ptransp0(x[..., :Split[0]], u[..., :Split[0]], c)
        euc = self.Euclidean.ptransp0(x[..., Split[0] : Split[1]], u[..., Split[0] : Split[1]], c)
        poin = self.Poincare.ptransp0(x[..., Split[1] : Split[2]], u[..., Split[1] : Split[2]], c)
        return torch.cat([hyper, euc, poin], dim = 1)


    def normalize(self, p):
        self.rescale_dims(p)
        return torch.cat([p[...,:Split[0]], self.Euclidean.normalize(p[...,Split[0]:Split[1]]), p[...,Split[1]:Split[2]]], dim=1)

    # def to_poincare(self, x, c):
    #     K = 1. / c
    #     sqrtK = K ** 0.5
    #     d = x.size(-1) - 1
    #     return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)
