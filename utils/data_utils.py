from pathlib import Path
import xml.etree.ElementTree as ET
from os import listdir, makedirs
from os.path import exists, isfile, isdir, join
import random
import numpy as np
from scipy.spatial import Delaunay


def get_subfolders(path):
    return [name for name in listdir(path) if isdir(join(path, name))]


def get_filenames(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def get_data_from_xml(root):
    img = root.find('image').text
    category = root.find('category').text
    bndbox = root.find('visible_bounds').attrib
    for target in bndbox:
        bndbox[target] = float(bndbox[target])
    keypoints = root.find('keypoints').findall('keypoint')
    keypoints = [keypoint.attrib for keypoint in keypoints]
    for keypoint in keypoints:
        x, y = float(keypoint['x']), float(keypoint['y'])
        for attr_drop in ['visible', 'x', 'y', 'z']:
            keypoint.pop(attr_drop, None)
        keypoint['pos'] = (x, y)
    return {'img': img, 'category': category, 'bndbox': bndbox, 'keypoints': keypoints}


def get_data_from_file(path):
    return get_data_from_xml(ET.parse(path).getroot())


def get_files_per_class(path):
    probs = {}
    files = {}
    for folder in get_subfolders(path):
        files[folder] = get_filenames(join(path, folder))
        probs[folder] = len(files[folder])
    total_count = sum([val for _, val in probs.items()])
    for key in probs.keys():
        probs[key] /= total_count
    return files, probs


def generate_data_from_dict(files_per_class, path):
    total_data = {}
    for folder in files_per_class.keys():
        total_data[folder] = []
        for file in files_per_class[folder]:
            chosen_file = join(path, folder, file)
            entry = get_data_from_file(chosen_file)
            if len(entry['keypoints']) > 2:
                total_data[folder].append(entry)
    return total_data


def generate_data(path):
    files, probs = get_files_per_class(path)
    return generate_data_from_dict(files, path), probs


def delaunay_tri(keypoints):
    points = [entry['pos'] for entry in keypoints]
    points = np.array(points)
    try:
        tri = Delaunay(points)
    except:
        tri = None
    return points, tri


def delaunay_to_adjacency(points, tri):
    num_points, _ = points.shape
    simplices = tri.simplices
    _, cols = simplices.shape
    simplices_flattened = simplices.reshape(-1,)
    reps = np.repeat(simplices_flattened, repeats=cols)
    tiles = np.tile(simplices, reps=cols).reshape(-1,)
    res = np.zeros((num_points, num_points))
    res[reps, tiles] = 1
    np.fill_diagonal(res, 0)
    return res


def make_graph(keypoints):
    points, tri = delaunay_tri(keypoints)
    graph = None if tri is None else delaunay_to_adjacency(points, tri)
    return points, tri, graph


def generate_training_data_for_class(class_data, start_id=0, max_samples=None):
    running_id = start_id
    resulting_data = []
    samples_per_class = len(class_data)
    counter = 0
    for id_source in range(samples_per_class - 1):
        for id_target in range(id_source + 1, samples_per_class):
            entry_s, entry_t = class_data[id_source], class_data[id_target]
            kp_s = [kp['name'] for kp in entry_s['keypoints']]
            kp_t = [kp['name'] for kp in entry_t['keypoints']]
            kp_common = [kp for kp in kp_s if kp in kp_t]
            if len(kp_common) > 2:
                corresp_kp_s = [kp_dic for kp_dic in entry_s['keypoints'] if kp_dic['name'] in kp_common]
                corresp_kp_t = [kp_dic for kp_dic in entry_t['keypoints'] if kp_dic['name'] in kp_common]
                points_s, _, graph_s = make_graph(corresp_kp_s)
                points_t, _, graph_t = make_graph(corresp_kp_t)
                # graph_s == None or graph_t == None => Delaunay failed
                if graph_s is not None and graph_t is not None:
                    resulting_data.append({'id': running_id,
                                           'img_s': entry_s['img'],
                                           'img_t': entry_t['img'],
                                           'cat': entry_s['category'],
                                           'kps': kp_common,
                                           'bndbox_s': [val for _, val in entry_s['bndbox'].items()],
                                           'bndbox_t': [val for _, val in entry_t['bndbox'].items()],
                                           'points_s': points_s,
                                           'points_t': points_t,
                                           'graph_s': graph_s,
                                           'graph_t': graph_t
                                          })
                    running_id += 1
                    counter += 1
                    if max_samples is not None and counter >= max_samples:
                        return resulting_data, running_id
    return resulting_data, running_id


def generate_data_according_to_prob(data, prob, max_samples=None):
    running_id = 0
    semantic_classes = list(data.keys())
    if max_samples is not None:
        max_samples = [int(p * max_samples) for _, p in prob.items()]
    else:
        max_samples = [None] * len(semantic_classes)
    result_total = []
    for idx, semantic_class in enumerate(semantic_classes):
        res_class, rid = generate_training_data_for_class(data[semantic_class],
                                                          start_id=running_id,
                                                          max_samples=max_samples[idx])
        running_id = rid
        result_total.append(res_class)
    return [item for sublist in result_total for item in sublist]


def generate_training_data(annotation_path, max_samples=None):
    data, prob = generate_data(annotation_path)
    return generate_data_according_to_prob(data, prob, max_samples=max_samples)


def make_graph_dirs(b_path, graph_dir):
    graph_dir = join(b_path, graph_dir)
    sub_dirs = ['adj', 'points']
    for s_dir in sub_dirs:
        sub_dir = join(graph_dir, s_dir)
        if not exists(sub_dir):
            makedirs(sub_dir)


def write_data_to_files(data_total, graph_path):
    random.shuffle(data_total)
    n_total = len(data_total)
    n_train = int(0.8 * n_total)
    data = {}
    data['train'] = data_total[:n_train]
    data['val'] = data_total[n_train:]
    out_files = {}
    out_files['train'] = open(join(graph_path, 'train.txt'), 'a')
    out_files['val'] = open(join(graph_path, 'val.txt'), 'a')
    for mode in data.keys():
        for entry in data[mode]:
            row = [str(entry['id']),
                   entry['img_s'],
                   entry['img_t'],
                   entry['cat'],
                   *[str(val) for val in entry['bndbox_s']],
                   *[str(val) for val in entry['bndbox_t']],
                   *[kp for kp in entry['kps']]
                   ]
            out_files[mode].write('\t'.join(row))
            out_files[mode].write('\n')
            # Write serialized numpy objects
            points_s, points_t = entry['points_s'], entry['points_t']
            points = np.stack((points_s, points_t), axis=0)
            np.save(join(graph_path, 'points', str(entry['id'])), points)
            graph_s, graph_t = entry['graph_s'], entry['graph_t']
            graph = np.stack((graph_s, graph_t), axis=0)
            np.save(join(graph_path, 'adj', str(entry['id'])), graph)
    out_files['train'].close()
    out_files['val'].close()