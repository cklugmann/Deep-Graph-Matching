import torch
import torch.nn as nn


class VotingLayer(nn.Module):
    def __init__(self, num_iterations=10, num_nodes=16):
        super(VotingLayer, self).__init__()
        self.num_iterations = num_iterations
        self.num_nodes = num_nodes

    def forward(self, v):
        batch_size = v.shape[0]
        S = v.reshape((batch_size, self.num_nodes, self.num_nodes)).permute(0, 2, 1)
        S = S + 10e-7
        for _ in range(self.num_iterations):
            A = torch.ones((batch_size, 1, self.num_nodes), device=S.device, dtype=S.dtype)
            tmp1 = torch.diag_embed(torch.bmm(A, S).squeeze(1)).permute(0, 2, 1)
            S_next, _ = torch.solve(S.permute(0, 2, 1), tmp1)
            S_next = S_next.permute(0, 2, 1)
            B = torch.ones((batch_size, self.num_nodes, 1), device=S.device, dtype=S.dtype)
            tmp2 = torch.diag_embed(torch.bmm(S_next, B).squeeze(2))
            S, _ = torch.solve(S_next, tmp2)

        return S


class DistancePredictor(nn.Module):
    def __init__(self, alpha=200.):
        super(DistancePredictor, self).__init__()
        self.alpha = alpha

    def forward(self, S, points_s, points_t):
        S = self.alpha * S
        sums = torch.sum(S, dim=2).unsqueeze(2)
        S = S/sums
        pred_t = torch.bmm(S, points_t)
        distance = pred_t - points_s
        return distance, pred_t


def distance_loss(d_pred, points_s, points_t, mask, eps=10e-6):
    """
    Computes the loss function as difference between distance vectors.
    :param d_pred: Predicted distance vector, tensor of shape (N, num_nodes, 2).
    :param mask: Mask encoding the true number of nodes, tensor of shape (N, num_nodes, 1)
    """
    mask = mask.type(d_pred.dtype)
    d_gt = points_t - points_s
    squared_sums = torch.sum(mask * (d_pred - d_gt) ** 2, dim=2)
    robust_norm = torch.sqrt(squared_sums + eps)
    return torch.sum(robust_norm) / robust_norm.shape[0]


if __name__ == '__main__':
    from torch.autograd import Variable, gradcheck

    dist = DistancePredictor()
    points_s = [[[1., 0.5], [0.1, 0.5]]]
    points_t = [[[0.5, 1.], [0.2, 0.6]]]
    d_pred = [[[0.25, 0.5], [0.25, 0.5]]]
    d_pred = Variable(torch.tensor(d_pred, dtype=torch.double), requires_grad=True)
    points_s = Variable(torch.tensor(points_s, dtype=torch.double), requires_grad=True)
    points_t = Variable(torch.tensor(points_t, dtype=torch.double), requires_grad=True)
    mask = [1, 1]
    mask = torch.tensor(mask, dtype=torch.int32).reshape(1, 2, 1)
    loss = distance_loss(d_pred, points_s, points_t, mask)
    loss.backward()
    print(loss.item())
    print(points_s.grad)
    print(points_t.grad)
    test = gradcheck(distance_loss, (d_pred, points_s, points_t, mask), eps=1e-6, atol=1e-4)
    print(test)