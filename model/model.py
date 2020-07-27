import os

import torch
import torch.nn as nn

from model.modules.feature_extractor import FeatureExtractor
from model.modules.affinity_matrix import AffinityLayer
from model.modules.power_iteration import PowerIteration
from model.modules.voting_layer import VotingLayer, DistancePredictor, distance_loss


def save_weights(model, save_path='.', filename_base={}):
    filename_base.setdefault('ext', 'extractor.pth')
    filename_base.setdefault('aff', 'affinity.pth')
    torch.save(model.feature_extractor.state_dict(), os.path.join(save_path, filename_base['ext']))
    torch.save(model.affinity.state_dict(), os.path.join(save_path, filename_base['aff']))


def load_weights(model, save_path='.', filename_base={}):
    filename_base.setdefault('ext', 'extractor.pth')
    filename_base.setdefault('aff', 'affinity.pth')
    model.feature_extractor.load_state_dict(torch.load(os.path.join(save_path, filename_base['ext'])))
    model.affinity.load_state_dict(torch.load(os.path.join(save_path, filename_base['aff'])))


class GMN(nn.Module):
    def __init__(self, node_pad, model_cfg):
        super(GMN, self).__init__()
        self.model_cfg = model_cfg
        self.feature_extractor = FeatureExtractor(node_pad)
        self.affinity = AffinityLayer(internal_dim=model_cfg.internal_dim)
        self.powerIterate = PowerIteration(num_iterations=model_cfg.power_iterations)
        self.voting = VotingLayer(num_iterations=model_cfg.sinkhorn_iterations, num_nodes=node_pad)
        self.distance = DistancePredictor(alpha=model_cfg.alpha)

    def forward(self, tuple):
        img1, img2 = tuple[:2]
        mask = tuple[-1]
        F1, F2, U1, U2 = self.feature_extractor(img1, img2, *tuple[-3:-1], mask)
        if self.model_cfg.fixed_features:
            F1, F2, U1, U2 = F1.detach(), F2.detach(), U1.detach(), U2.detach()
        M = self.affinity(F1, F2, U1, U2, *tuple[2:6], mask)
        v = self.powerIterate(M)
        S = self.voting(v)
        dist_pred, pos_red = self.distance(S, *tuple[-3:-1])
        return dist_pred, pos_red