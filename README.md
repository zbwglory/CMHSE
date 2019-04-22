# Cross-Modal and Hierarchical Modeling of Video and Text
The code repository for "[Cross-Modal and Hierarchical Modeling of Video and Text](https://arxiv.org/abs/1810.07212)" in PyTorch

### Prerequisites

The following packages are required to run the scripts:

- [PyTorch >= 0.4 and torchvision](https://pytorch.org)

- Package [tensorboardX](https://github.com/lanpa/tensorboardX) and NLTK

- Dataset: please download [features](https://drive.google.com/drive/folders/1341zliZg8-kveVFqRIgmreG8re_JcoUy?usp=sharing) (The feature is still uploading.) and put them into the folder data/anet_precomp and data/didemo_precomp


### Model Evaluation

The learned model on ActivityNet and DiDeMo can be found in [this link](https://drive.google.com/file/d/1ELrUGE315JEOKudNeCh42gpXkGFx4vBd/view?usp=sharing). You can run **train.py** with option **--resume** and **--eval_only** to evaluate a given model, with options similar to the training scripts as below. 

For a model with Inception feature on ActivityNet dataset at "./runs/release/activitynet/ICEP/hse_tau5e-4/run1/checkpoint.pth.tar", it can be evaluated by:

    $ python train.py anet_precomp --feat_name icep --img_dim 2048 --resume ./runs/release/activitynet/ICEP/hse_tau5e-4/run1/checkpoint.pth.tar --eval_only

For a model with C3D feature on ActivityNet dataset at "./runs/release/activitynet/C3D/hse_tau5e-4/run1/checkpoint.pth.tar", it can be evaluated by:

    $ python train.py anet_precomp --feat_name c3d --img_dim 500 --resume ./runs/release/activitynet/C3D/hse_tau5e-4/run1/checkpoint.pth.tar --eval_only

We presume the input model is a GPU stored model.

### Model Training

To reproduce our experiments with HSE, please use **train.py** and follow the instructions below. To train HSE with \tau=5e-4, please with 

    $ --reconstruct_loss --lowest_reconstruct_loss
    
For example, to train HSE with \tau=5e-4 on ActivityNet with C3D feature:

    $ python train.py anet_precomp --feat_name c3d --img_dim 500 --low_level_loss --reconstruct_loss --lowest_reconstruct_loss --norm
    
To train HSE with \tau=5e-4 on ActivityNet with Inception feature (The feature is still being uploaded):

    $ python train.py anet_precomp --feat_name icep --img_dim 2048 --low_level_loss --reconstruct_loss --lowest_reconstruct_loss --norm
    
To train HSE with \tau=5e-4 on Didemo with Inception feature:

    $ python train.py didemo_precomp --feat_name icep --img_dim 2048 --low_level_loss --reconstruct_loss --lowest_reconstruct_loss --norm
    
To train HSE with \tau=0 on ActivityNet with C3D feature:

    $ python train.py anet_precomp --feat_name c3d --img_dim 500 --low_level_loss --norm


## .bib citation
If this repo helps in your work, please cite the following paper:

    @inproceedings{DBLP:conf/eccv/ZhangHS18,
      author    = {Bowen Zhang and
               Hexiang Hu and
               Fei Sha},
      title     = {Cross-Modal and Hierarchical Modeling of Video and Text},
      booktitle = {Computer Vision - {ECCV} 2018 - 15th European Conference, Munich,
               Germany, September 8-14, 2018, Proceedings, Part {XIII}},
      pages     = {385--401},
      year      = {2018}
}


## Acknowledgment
We thank following repos providing helpful components/functions in our work.

- [VSE++](https://github.com/fartashf/vsepp) for the framework
- [TSN](https://github.com/yjxiong/temporal-segment-networks) for the inception-v3 feature

## Contacts
Please report bugs and errors to
  
    Bowen Zhang: zbwglory@gmail.com
