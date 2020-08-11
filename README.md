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
5. (Optional) You may want to change the settings for the model and training procedure.

## Configuration files

Most of the adjustments to the training or the model to be trained can be made using the configuration files that can be found in the `configs` directory.

- `model_config.json`
 * `fixed_features`: set this to true or false, depending on whether the weights of the pretrained feature extractor should be fixed or not
 * `power_iterations`: maximum number of iterations in the power iteration layer
 * `sinkhorn_iterations`: maximum number of iterations in the bi-stochastic layer
 * `alpha`: a hyperparameter used to weight the possible target positions according to the bi-stochastic matrix
 * `internal_dim`: this is the number of channels of feature maps to be used to extract node-based features (note: it is necessary to set this value but changing it will not guarantee consistency since layers at which the features are extracted are currently harcoded)
- `train_config.json
 * `batch_size_train`: size of the training batches
 * `optimizer`: some self-explanatory hyper parameters for the Adam optimizer
 * `batch_size_val`: size of the validation batches
 * `num_images_to_print`: number of images to save every once in a while (must be less than or equal to validation batch size, otherwise this value is clamped)
 * `iters_per_epoch`: number of batches to be processed per epoch
 * `experiments_base_path`: base path where all the results of the experiments are written to
 * `save_weights_path`: location where the checkpoints shall be saved

