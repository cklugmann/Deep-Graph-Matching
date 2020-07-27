from functools import reduce

import numpy as np
import torch


def flattened_size(shape):
    return reduce(lambda x, y: x * y, shape)


def flattened_batchwise(shape):
    N = shape[0]
    flattened = flattened_size(shape[1:])
    return (N, flattened)


def kron(a, b):
    """
    (Credits to https://gist.github.com/yulkang)
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def extract_diag(A):
    return torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))


def diag_batchwise(A):
    flattened = A.reshape(flattened_batchwise(A.shape))
    batch_size, num_elements = flattened.shape
    sparse_matrices = []
    spatial_indices = np.arange(num_elements).tolist()
    for n in range(batch_size):
        vals = A[n].t().reshape(-1,)
        indices = torch.LongTensor([spatial_indices, spatial_indices])
        indices = indices.cuda(vals.get_device()) if vals.is_cuda else indices
        # if torch.cuda.is_available():
        #    indices = indices.cuda()
        sparse_res = torch.sparse.FloatTensor(indices, vals, torch.Size((num_elements, num_elements)))
        sparse_matrices.append(sparse_res)
    return sparse_matrices


def diag_batchwise_dense(A):
    flattened = flatten_column_major(A)
    return torch.diag_embed(flattened)


def to_list_of_tensor(A):
    batch_size = A.shape[0]
    res = [A[n] for n in range(batch_size)]
    return res


def bmm_sparse(A, B):
    batch_size = len(A)
    C = [torch.sparse.mm(A[n], B[n]) for n in range(batch_size)]
    return C


def flatten_column_major(A):
    """
    Flatten a batch of matrices according to fortran-style ordering.
    :param A: A tensor of shape (N, D1, D2)
    :return: A flattened view of this batch of matrices of shape (N, D1*D2)
    """
    return A.permute(0, 2, 1).reshape(flattened_batchwise(A.shape))