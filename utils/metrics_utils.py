import torch


def pck_metric(points_t, points_pred, mask, ori_size=(256, 256), alpha=0.1):
    num_nodes = torch.sum(mask, dim=1).squeeze(-1)
    dists = torch.norm(mask * (points_t - points_pred), dim=2)
    h, w = ori_size
    threshold = h ** 2 + w ** 2
    threshold = torch.tensor(threshold, dtype=points_t.dtype, device=points_t.device)
    threshold = alpha * torch.sqrt(threshold)
    cnt = 0
    n_total = 0
    for idx, n_nodes in enumerate(num_nodes):
        d = dists[idx, :n_nodes]
        tmp = torch.zeros_like(d)
        tmp[d < threshold] = 1.
        cnt += torch.sum(tmp).item()
        n_total += n_nodes.item()
    return cnt / n_total


if __name__ == '__main__':
    points_t = [[[0.5, 1.], [0.2, 0.6]],
                [[0.5, 1.], [0.2, 0.6]]]
    points_pred = [[[0.5, 1.], [0.25, 0.5]],
                   [[0.3, 1.], [0.2, 0.6]]]
    points_pred = torch.tensor(points_pred, dtype=torch.double)
    points_t = torch.tensor(points_t, dtype=torch.double)
    mask = [[1, 1], [1, 1]]
    mask = torch.tensor(mask, dtype=torch.int32).reshape(2, 2, 1)
    p = pck_metric(points_t, points_pred, mask, ori_size=(1., 1.))
    print(p)