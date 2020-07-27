import torch
import torch.nn as nn
import torchvision.models.vgg as models

"""
    Note: the functions feature_align, interp_2d, bilinear_interpolate_torch are basically
    taken from the code of Wang, Runzhong and Yan, Junchi and Yang, Xiaokang.
"""


def feature_align(raw_feature, P, mask):
    """
    Perform feature align from the raw feature map.
    :param raw_feature: raw feature map
    :param P: point set containing point coordinates
    :return: F
    """
    device = raw_feature.device
    ori_size = (256, 256)
    ns_t = torch.sum(mask, dim=1)
    batch_num = raw_feature.shape[0]
    channel_num = raw_feature.shape[1]
    n_max = P.shape[1]

    ori_size = torch.tensor(ori_size, dtype=torch.float32, device=device)
    F = torch.zeros(batch_num, channel_num, n_max, dtype=torch.float32, device=device)
    for idx, feature in enumerate(raw_feature):
        n = ns_t[idx]
        feat_size = torch.as_tensor(feature.shape[1:3], dtype=torch.float32, device=device)
        _P = P[idx, 0:n]
        F[idx, :, 0:n] = interp_2d(feature, _P, ori_size, feat_size)
    return F


def interp_2d(z, P, ori_size, feat_size):
    """
    Interpolate in 2d grid space. z can be 3-dimensional where the 3rd dimension is feature vector.
    :param z: 2d/3d feature map
    :param P: input point set
    :param feat_size: size of the feature map
    :return: F
    """
    device = z.device

    step = ori_size / feat_size
    out = torch.zeros(z.shape[0], P.shape[0], dtype=torch.float32, device=device)
    for i, p in enumerate(P):
        p = (p - step / 2) / ori_size * feat_size
        out[:, i] = bilinear_interpolate_torch(z, p[0], p[1])

    return out


def bilinear_interpolate_torch(im, x, y):
    """
    Bi-linear interpolate 3d feature map im to 2d plane (x, y)
    :param im: 3d feature map
    :param x: x coordinate
    :param y: y coordinate
    :return: interpolated feature vector
    """
    device = im.device
    x = x.to(im.dtype).to(device)
    y = y.to(im.dtype).to(device)

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[2] - 1)
    x1 = torch.clamp(x1, 0, im.shape[2] - 1)
    y0 = torch.clamp(y0, 0, im.shape[1] - 1)
    y1 = torch.clamp(y1, 0, im.shape[1] - 1)

    x0 = x0.to(torch.int32).to(device)
    x1 = x1.to(torch.int32).to(device)
    y0 = y0.to(torch.int32).to(device)
    y1 = y1.to(torch.int32).to(device)

    Ia = im[:, y0, x0]
    Ib = im[:, y1, x0]
    Ic = im[:, y0, x1]
    Id = im[:, y1, x1]

    # to perform nearest neighbor interpolation if out of bounds
    if x0 == x1:
        if x0 == 0:
            x0 -= 1
        else:
            x1 += 1
    if y0 == y1:
        if y0 == 0:
            y0 -= 1
        else:
            y1 += 1

    x0 = x0.to(im.dtype).to(device)
    x1 = x1.to(im.dtype).to(device)
    y0 = y0.to(im.dtype).to(device)
    y1 = y1.to(im.dtype).to(device)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out


class FeatureVGG(nn.Module):
    def __init__(self, num_nodes):
        super(FeatureVGG, self).__init__()
        # We need VGG16 features up to layer conv5_1
        self.vgg16_features = models.vgg16(pretrained=True).features[:26]
        self.extract_at = [20, 25]
        self.num_nodes = num_nodes

    def forward(self, x, points, mask):
        features = []
        for idx, layer in enumerate(self.vgg16_features):
            x = layer(x)
            if idx in self.extract_at:
                features.append(x)
        F_features, U_features = features[0], features[1]

        # locResp = nn.LocalResponseNorm(F_features.shape[1] * 2, alpha=F_features.shape[1] * 2, beta=0.5, k=1.)
        # F_features, U_features = locResp(F_features), locResp(U_features)

        F_features = feature_align(F_features, points, mask)
        U_features = feature_align(U_features, points, mask)
        F_features = F_features.permute(0, 2, 1)
        U_features = U_features.permute(0, 2, 1)
        return F_features, U_features


class FeatureExtractor(nn.Module):
    def __init__(self, num_nodes):
        super(FeatureExtractor, self).__init__()
        self.feature_vgg = FeatureVGG(num_nodes)

    def forward(self, img1, img2, points1, points2, mask):
        F1, U1 = self.feature_vgg(img1, points1, mask)
        F2, U2 = self.feature_vgg(img2, points2, mask)
        return F1, F2, U1, U2

