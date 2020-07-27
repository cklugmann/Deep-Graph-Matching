# Deep Learning of Graph Matching - Final Project

This repository contains the code for my final project for the course 'Deep Vision' at the University of Heidelberg in summer term 2020.

The project is an implementation of the paper "Deep learning of graph matching" by Zanfir and Sminchisescu:

```
@inproceedings{zanfir2018deep,
  title={Deep learning of graph matching},
  author={Zanfir, Andrei and Sminchisescu, Cristian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2684--2693},
  year={2018}
}
```

## Quick installation instructions

1. Download [Pascal VOC 2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html).
2. Download the [keypoint annotations](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) from Berkeley and extract them within the VOC dataset folder.
3. To generate the graph structures and form image pairs, refer to the code base and follow the instructions in the notebook `notebooks/explore_data.ipynb`.
4. Change the entries in the configuration file `configs/data_config.json` accordingly (you do not need to touch `node_pad` or `edge_pad` there).
5. (Optional) You may want to change the settings for the model and training procedure. A description of the different settings will follow shortly.

