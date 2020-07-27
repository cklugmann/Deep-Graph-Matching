import os
import torch

from utils.config_utils import get_config_from_json

from model.model import GMN
from trainer.trainer import Trainer


def main(conf):
    data_conf = conf['data']
    model_conf = conf['model']
    train_conf = conf['train']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gmn = GMN(data_conf.node_pad, model_conf)
    gmn.to(device)
    trainer = Trainer(gmn, train_conf, data_conf)
    trainer.train_epochs(number_epochs=16)

if __name__ == '__main__':
    data_conf, _ = get_config_from_json(os.path.join('../configs', 'data_config.json'))
    model_conf, _ = get_config_from_json(os.path.join('../configs', 'model_config.json'))
    train_conf, _ = get_config_from_json(os.path.join('../configs', 'train_config.json'))
    conf = {'data': data_conf,
            'model': model_conf,
            'train': train_conf}

    main(conf)