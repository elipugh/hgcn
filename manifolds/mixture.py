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
        self.Split = [0,0]

    def sqdist(self, x, y, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.sqdist(x, y, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.sqdist(x, y, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.sqdist(x, y, c)
        self.rescale_dims(x)
        if self.Fractions[0] != 0:
            hyper = self.Hyperboloid.sqdist(x[:,:self.Split[0]], y[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.sqdist(x[:,self.Split[0]:self.Split[1]], y[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.sqdist(x[:,self.Split[1]:], x[:,self.Split[1]:], c)
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
            hyper = self.Hyperboloid.proj(x[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.proj(x[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.proj(x[:,self.Split[1]:], c)
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
            hyper = self.Hyperboloid.proj_tan(u[:,:self.Split[0]], x[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.proj_tan(u[:,self.Split[0]:self.Split[1]], x[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.proj_tan(u[:,self.Split[1]:], x[:,self.Split[1]:], c)
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
            hyper = self.Hyperboloid.proj_tan0(u[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.proj_tan0(u[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.proj_tan0(u[:,self.Split[1]:], c)
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
            hyper = self.Hyperboloid.expmap(u[:,:self.Split[0]], x[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.expmap(u[:,self.Split[0]:self.Split[1]], x[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.expmap(u[:,self.Split[1]:], x[:,self.Split[1]:], c)
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
            hyper = self.Hyperboloid.logmap(x[:,:self.Split[0]], y[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.logmap(x[:, self.Split[0]:self.Split[1]], y[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.logmap(x[:, self.Split[1]:], y[:,self.Split[1]:], c)
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
            hyper = self.Hyperboloid.expmap0(u[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.expmap0(u[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.expmap0(u[:,self.Split[1]:], c)
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
            hyper = self.Hyperboloid.logmap0(x[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.logmap0(x[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.logmap0(x[:,self.Split[1]:], c)
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
            hyper = self.Hyperboloid.mobius_add(x[:,:self.Split[0]], y[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.mobius_add(x[:,self.Split[0]:self.Split[1]], y[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.mobius_add(x[:,self.Split[1]:], y[:,self.Split[1]:], c)
        if self.Fractions[0] == 0:
            return torch.cat([euc, poin], dim = 1)
        if self.Fractions[1] == 0:
            return torch.cat([hyper, poin], dim = 1)
        if self.Fractions[2] == 0:
            return torch.cat([hyper, euc], dim = 1)
        return torch.cat([hyper, euc, poin], dim = 1)

    def old_mobius_matvec1(self,m,x,c):
        return self.expmap0( self.logmap0(x,c) @ m.transpose(-2,-1), c )

    def old_mobius_matvec2(self,m,x,c):
        self.rescale_dims(x)
        add = None
        if self.Fractions[0] != 0:
            hmu = self.Hyperboloid.mobius_matvec(
              m[:,:self.Split[0]],
              x[:,:self.Split[0]],
              c
            )
            add = hmu
        if self.Fractions[1] != 0:
            emu = self.Euclidean.mobius_matvec(
              m[:,self.Split[0]:self.Split[1]],
              x[:,self.Split[0]:self.Split[1]],
              c
            )
            if add is None:
                add = emu
            else:
                add = self.mobius_add(add,emu,c)
        if self.Fractions[2] != 0:
            pmu = self.Poincare.mobius_matvec(
              m[:,self.Split[1]:],
              x[:,self.Split[1]:],
              c
            )
            if add is None:
                add = pmu
            else:
                add = self.mobius_add(add,pmu,c)
        return add

    def old_mobius_matvec3(self,m,x,c):
        add = None
        if self.Fractions[0] != 0:
            self.rescale_dims(x)
            m1 = m[:,:self.Split[0]]
            x1 = x[:,:self.Split[0]]
            self.rescale_dims(m1.transpose(-2,-1))
            m1 = m1[:self.Split[0]]
            hmu = self.Hyperboloid.mobius_matvec(m1, x1, c)
            add = hmu
        if self.Fractions[1] != 0:
            self.rescale_dims(x)
            m2 = m[:,self.Split[0]:self.Split[1]]
            x2 = x[:,self.Split[0]:self.Split[1]]
            self.rescale_dims(m2.transpose(-2,-1))
            m2 = m2[self.Split[0]:self.Split[1]]
            emu = self.Euclidean.mobius_matvec(m2, x2, c)
            if add is None:
              add = emu
            else:
              add = torch.cat([add,emu], dim=1)
        if self.Fractions[2] != 0:
            self.rescale_dims(x)
            m3 = m[:,self.Split[1]:]
            x3 = x[:,self.Split[1]:]
            self.rescale_dims(m3.transpose(-2,-1))
            m3 = m3[self.Split[1]:]
            pmu = self.Poincare.mobius_matvec(m3, x3, c)
            if add is None:
              add = pmu
            else:
              add = torch.cat([add,pmu], dim=1)
        return add

    def mobius_matvec(self, m, x, c):
        if (self.Fractions[1]==0) and (self.Fractions[2]==0):
            return self.Hyperboloid.mobius_matvec(m, x, c)
        if (self.Fractions[0]==0) and (self.Fractions[2]==0):
            return self.Euclidean.mobius_matvec(m, x, c)
        if (self.Fractions[0]==0) and (self.Fractions[1]==0):
            return self.Poincare.mobius_matvec(m, x, c)
        self.rescale_dims(x)
        # hyperboloid
        if self.Fractions[0] != 0:
            hyper_u = self.Hyperboloid.logmap0(x[:,:self.Split[0]], c)
            accumulated_u = torch.cat([hyper_u, x[:,self.Split[0]:]], dim=1)
        else:
            accumulated_u = x
            
        if self.Fractions[2] != 0:
            # poincare
            sqrt_c = c ** 0.5
            poinc_x_norm = x[:,self.Split[1]:].norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)

        # multiply
        mu = accumulated_u @ m.transpose(-1, -2)
        self.rescale_dims(mu)

        # hyperboloid
        if self.Fractions[0] != 0:
            hyper_final = self.Hyperboloid.expmap0(mu[:,:self.Split[0]], c)
        if self.Fractions[2] != 0:
            poinc_mx_norm = mu[:,self.Split[1]:].norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
            res_c = tanh(poinc_mx_norm/poinc_x_norm*artanh(sqrt_c*poinc_x_norm))*mu[:,self.Split[1]:]/(poinc_mx_norm*sqrt_c)
            cond = (mu[:,self.Split[1]:] == 0).prod(-1, keepdim=True, dtype=torch.uint8)
            res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
            poincare_final = torch.where(cond, res_0, res_c)

        if self.Fractions[0] == 0:
            new_mu = torch.cat([mu[:,self.Split[0]:self.Split[1]], poincare_final], dim=1)
        elif self.Fractions[1] == 0:
            new_mu = torch.cat([hyper_final, poincare_final], dim=1)
        elif self.Fractions[2] == 0:
            new_mu = torch.cat([hyper_final, mu[:,self.Split[0]:self.Split[1]]], dim=1)
        else:
            new_mu = torch.cat([hyper_final, mu[:,self.Split[0]:self.Split[1]], poincare_final], dim=1)

        return new_mu


    def rescale_dims(self, x):
        length = len(x[0,:])
        if length != self.length:
            self.Split[0] = int(self.Fractions[0] * length)
            if self.Fractions[2] == 0:
                self.Split[1] = length
            else:
                self.Split[1] = int(self.Fractions[1] * length) + self.Split[0]
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
            hyper = self.Hyperboloid.ptransp(x[:,:self.Split[0]], y[:,:self.Split[0]], u[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.ptransp(x[:,self.Split[0]:self.Split[1]], y[:,self.Split[0]:self.Split[1]], u[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.ptransp(x[:,self.Split[1]:], y[:,self.Split[1]:], u[:,self.Split[1]:], c)
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
            hyper = self.Hyperboloid.ptransp0(x[:,:self.Split[0]], u[:,:self.Split[0]], c)
        if self.Fractions[1] != 0:
            euc = self.Euclidean.ptransp0(x[:,self.Split[0]:self.Split[1]], u[:,self.Split[0]:self.Split[1]], c)
        if self.Fractions[2] != 0:
            poin = self.Poincare.ptransp0(x[:,self.Split[1]:], u[:,self.Split[1]:], c)
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
            hyper = p[:,:self.Split[0]]
        if self.Fractions[1] != 0:
            euc = self.Euclidean.normalize(p[:,self.Split[0]:self.Split[1]])
        if self.Fractions[2] != 0:
            poin = p[:,self.Split[1]:]
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


