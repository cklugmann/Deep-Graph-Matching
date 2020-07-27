import torch
import torch.nn as nn
from torch.autograd import Variable


class PowerIteration(nn.Module):
    def __init__(self, num_iterations=10):
        super(PowerIteration, self).__init__()
        self.num_iterations = num_iterations

    def forward(self, M):
        batch_size, n_pairs = M.shape[:2]
        v = Variable(torch.ones((batch_size, n_pairs, 1), device=M.device).type(M.dtype), requires_grad=False)
        vpre = v
        for idx in range(self.num_iterations):
            v = torch.bmm(M, v)
            v_norm = torch.norm(v, dim=1).unsqueeze(1)
            v = v / v_norm
            if torch.norm(v - vpre) < 10e-7:
                return v
            vpre = v
        # v = torch.clamp(v, 0.)
        return v

if __name__ == '__main__':
    from torch.autograd import gradcheck
    pi = PowerIteration(num_iterations=10)
    M = [[[0.3, .3, 0.1, 0.], [1., 0.2, 0.3, 0.5], [0.5, 1., 0.5, 0.3], [0.8, 0.4, 0.2, 0.8]]]
    M = torch.tensor(M, dtype=torch.double, requires_grad=True)
    test = gradcheck(pi, (M,), eps=1e-6, atol=1e-4)
    print(test)