import os
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image


def crop_args(bndbox):
    h, w, x, y = bndbox
    return (x, y, x+w, y+h)


def create_graph_matrices(adj, node_pad, edge_pad):
    num_nodes = adj.shape[0]
    G = np.zeros((node_pad, edge_pad), dtype=np.float32)
    H = np.zeros((node_pad, edge_pad), dtype=np.float32)
    edge_idx = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i, j] == 1:
                G[i, edge_idx] = 1
                H[j, edge_idx] = 1
                edge_idx += 1
    return G, H


class Dataset(data.Dataset):
    def __init__(self, data_config, mode='train', dtype=torch.float):
        self.data_config = data_config
        self.mode = mode if mode in ['train', 'val'] else 'train'
        self.dtype = dtype
        filename = os.path.join(self.data_config.base_path,
                                self.data_config.graph_folder,
                                '.'.join([mode, 'txt']))

        def process_entry(entry):
            entry = entry.rstrip().split('\t')
            bndbox_s = [float(val) for val in entry[4:8]]
            bndbox_t = [float(val) for val in entry[8:12]]
            return {'id': entry[0],
                    'img_s': entry[1],
                    'img_t': entry[2],
                    'cat': entry[3],
                    'bndbox_s': bndbox_s,
                    'bndbox_t': bndbox_t}

        with open(filename, 'r') as f:
            self.contents = list(map(process_entry, f.readlines()))

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        row = self.contents[index]

        # First, transform source and target image
        img_path = os.path.join(self.data_config.base_path,
                                self.data_config.img_folder)
        transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
             ])
        img_source = Image.open(os.path.join(img_path, '.'.join([row['img_s'], 'jpg'])))
        img_target = Image.open(os.path.join(img_path, '.'.join([row['img_t'], 'jpg'])))
        img_source = img_source.crop(crop_args(row['bndbox_s']))
        img_target = img_target.crop(crop_args(row['bndbox_t']))

        img_source = transform(img_source).type(dtype=self.dtype)
        img_target = transform(img_target).type(dtype=self.dtype)

        # Load matrices from serialized numpy objects
        graph_path = os.path.join(self.data_config.base_path,
                                  self.data_config.graph_folder)
        adj_file = os.path.join(graph_path, 'adj', '.'.join([row['id'], 'npy']))
        points_file = os.path.join(graph_path, 'points', '.'.join([row['id'], 'npy']))

        adj = np.load(adj_file).astype(np.float32)
        points = np.load(points_file).astype(np.float32)
        adj_s, adj_t = adj[0], adj[1]
        points_s, points_t = points[0], points[1]

        # We transform the points to [0,1]x[0,1]
        h_s, w_s, x_s, y_s = row['bndbox_s']
        h_t, w_t, x_t, y_t = row['bndbox_t']
        points_s = np.maximum(points_s - np.array([x_s, y_s]), np.zeros_like(points_s))
        points_t = np.maximum(points_t - np.array([x_t, y_t]), np.zeros_like(points_t))
        points_s[:, 0] *= 256./w_s
        points_s[:, 1] *= 256./h_s
        points_t[:, 0] *= 256./w_t
        points_t[:, 1] *= 256./h_t

        G_s, H_s = create_graph_matrices(adj_s,
                                         node_pad=self.data_config.node_pad,
                                         edge_pad=self.data_config.edge_pad)
        G_t, H_t = create_graph_matrices(adj_t,
                                         node_pad=self.data_config.node_pad,
                                         edge_pad=self.data_config.edge_pad)

        G_s = torch.from_numpy(G_s)#.to_sparse()
        H_s = torch.from_numpy(H_s)#.to_sparse()
        G_t = torch.from_numpy(G_t)#.to_sparse()
        H_t = torch.from_numpy(H_t)#.to_sparse()

        num_true_nodes = points_s.shape[0]
        pad = self.data_config.node_pad - num_true_nodes
        pad_zeros = np.zeros((pad, 2), dtype=points.dtype)
        padded_points_s = np.concatenate((points_s, pad_zeros)).astype(np.float32)
        padded_points_t = np.concatenate((points_t, pad_zeros)).astype(np.float32)
        padded_points_s = torch.from_numpy(padded_points_s)
        padded_points_t = torch.from_numpy(padded_points_t)

        # We use mask to keep track of the true (i.e. not padded) nodes
        mask = torch.zeros((self.data_config.node_pad, 1), dtype=torch.int32)
        mask[:num_true_nodes] = 1

        return img_source, img_target, G_s, G_t, H_s, H_t, padded_points_s, padded_points_t, mask