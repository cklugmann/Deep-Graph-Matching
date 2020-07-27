import math
import torch
import torch.nn as nn

from utils.tensor_utils import kron
from utils.tensor_utils import diag_batchwise, diag_batchwise_dense, bmm_sparse, to_list_of_tensor


def concat_features(features, G, H, num_nodes):
    """
    :param features: tensor of shape (pad_nodes, internal_dim)
    :return: Features for all edges
    """
    internal_dim = features.shape[1]
    edge_pad = G.shape[-1]
    edge_features = torch.zeros((edge_pad, 2 * internal_dim),
                                dtype=features.dtype,
                                device=features.device)
    edge_idx = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if G[i, edge_idx] == 1 and H[j, edge_idx] == 1:
                edge_features[edge_idx] = torch.cat([features[i], features[j]], 0)
                edge_idx += 1
    return edge_features


class AffinityLayer(nn.Module):
    def __init__(self, internal_dim=2048):
        super(AffinityLayer, self).__init__()
        self.internal_dim = internal_dim
        self.lambda1 = nn.Parameter(torch.zeros((internal_dim, internal_dim)))
        self.lambda2 = nn.Parameter(torch.zeros((internal_dim, internal_dim)))
        self.reset_parameters()

    def forward(self, F1, F2, U1, U2, G1, G2, H1, H2, mask):
        # We pass mask to obtain the true number of nodes
        num_nodes = torch.sum(mask, dim=1)
        batch_size = F1.shape[0]
        edge_features1 = []
        edge_features2 = []
        for n in range(batch_size):
            edge_features1.append(concat_features(F1[n], G1[n], H1[n], num_nodes[n]))
            edge_features2.append(concat_features(F2[n], G2[n], H2[n], num_nodes[n]))
        X = torch.stack(edge_features1)
        Y = torch.stack(edge_features2)
        la1, la2 = nn.functional.relu(self.lambda1), nn.functional.relu(self.lambda2)
        params = torch.cat((torch.cat((la1, la2), dim=1),
                            torch.cat((la2, la1), dim=1)),
                           dim=0)
        M_edge = torch.bmm(torch.matmul(X, params), Y.permute(0, 2, 1))
        M_node = torch.bmm(U1, U2.permute(0, 2, 1))

        tmp1 = diag_batchwise(M_edge)
        tmp2 = to_list_of_tensor(kron(H2, H1).permute(0, 2, 1))
        tmp3 = bmm_sparse(tmp1, tmp2)
        edge_part = torch.stack([torch.mm(kron(G2, G1)[i], tmp3[i]) for i in range(batch_size)])
        node_part = diag_batchwise_dense(M_node)
        M = node_part + edge_part
        return M

    def reset_parameters(self):
        """
        Initialization of parameters according to Wang et al.
        """
        stdv = 1. / math.sqrt(self.lambda1.shape[1] * 2)
        self.lambda1.data.uniform_(-1. * stdv, stdv)
        self.lambda2.data.uniform_(-1. * stdv, stdv)
        self.lambda1.data += torch.eye(self.internal_dim) / 2.
        self.lambda2.data += torch.eye(self.internal_dim) / 2.