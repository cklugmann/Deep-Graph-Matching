import os
from utils.config_utils import get_config_from_json


def test_config():
    config_file = os.path.join('../configs', 'test_config.json')
    data_config, _ = get_config_from_json(config_file)
    assert data_config.id == 42
    assert data_config.person['name'] == 'Bob'
    assert data_config.person['age'] == 30
