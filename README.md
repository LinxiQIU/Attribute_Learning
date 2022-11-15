# Attribute_Learning
This project is a sub-project of the research project **AgiProbot** from KIT and Bosch. We develop a benchmark including 2D synthetic image datasets and 3D synthetic point cloud datasets. In this part, we are going to use DGCNN as the backbone for encoding point-wise features and a global feature. We consider the classification-segmentation parallel training as a meta-block, note that since the one-hot attributes *T* and *N_s* indicate the validity of some other attributes, apart from the block of motor type classification, another block for classifying the number of cover screws has also been included in the meta-block. The regression tasks compose a second tail block.

<img src="https://github.com/LinxiQIU/Attribute_Learning/blob/main/images/mtl_reg.png" width="700" height="370">

## Environments Requirements

CUDA = 10.2

Python >= 3.7.0

PyTorch = 1.6

The mentioned API are the basic API. In the training process,if there is warning that some modul is missing. you could direct use pip install to install specific modul.

## Regression Evaluation Metrics
<img src="https://github.com/LinxiQIU/Attribute_Learning/blob/main/images/attr_metrics.jpg" width="400" height="500">

(1) **Size Relative Error (SRE)**. This metric evaluates the predicted overall motor size using four key attributes which represent the main body of motors: the lengths of the bottom and the sub-bottom part, pole pot, and the diameters of the gear regions.

(2) **Gear Location Error (GLE)**. This metric evaluates the distance error between the predicted location of gear region center and its ground truth. Involved attributes are the center point coordinate values.

(3) **Motor Rotation Error (MRE)**. This metric evaluates the absolute motor rotation error. Involved attributes are the motor rotations along three axes.

(4) **Screw Location Error (SLE)**. This metric evaluates the distance error between the predicted cover screw positions and their ground truth. Involved attributes are the screw position coordinate values.

## Training Methods
With the proposed architecture, there are several possible ways to train those blocks. 



(i) **Totally separate training**: the encoder and the meta-block are trained as one network first, and then another same-structure encoder and the regression block are trained as another totally separate network. No information is shared between two trainings. 

<img src="https://github.com/LinxiQIU/Attribute_Learning/blob/main/images/separate.png" width="600" height="320">

```python
CUDA_VISIBLE_DEVICES=0,1 python train_attr1.py --exp_name separate --change adawm_reg --root /home/ies/dataset/dataset1000 --epochs 200
```

Explanation of the important parameters:

* CUDA_VISIBLE_DEVICES: set the visible gpu

* main_cls.py: choose of which script will be run

* exp_name: the paremeter(separate) means the results of the separate training method

* change: give the information of a specific experiment (e.g. the changes of batch_size, epochs or optimizer)

* root: the root directory of training dataset


(ii) **Use meta-block for pre-training**: this method is similar to the last one, but the encoder weights in the second step will be initialized with the weights from the first step. The meta-block is used for pre-training.

<img src="https://github.com/LinxiQIU/Attribute_Learning/blob/main/images/pretrain.png" width="600" height="320">

```python
CUDA_VISIBLE_DEVICES=0,1 python train_attr_pretrain.py --exp_name pretrain1 --change adamw_no_seg --with_seg True --epochs 200 --root /home/ies/dataset/dataset1000
```

(iii) **Encoder-shared yet tail blocks trained in parallel**: the encoder is shared between two tail blocks and three blocks compose one joint network. All tasks are trained in parallel.

<img src="https://github.com/LinxiQIU/Attribute_Learning/blob/main/images/parallel.png" width="600" height="320">

```python
CUDA_VISIBLE_DEVICES=0,1 python train_attr_parallel.py --exp_name parallel --change adamw+wseg+5e-4 --with_seg True --epochs 200 --lr 0.0005--root /home/ies/dataset/dataset1000
```

(iv) **Encoder-shared and tail blocks trained iteratively**: in each training step, the encoder is firstly connected with the meta-block. We compute the loss, perform gradient back propagation and update model weights with these two blocks. Then the same encoder connects with the regression block in a switch manner, the input is reprocessed with the weight-updated encoder to get new encoded representations which is used for computing the regression loss. We then again perform gradient back propagation and weight update in these two blocks. This action performs iteratively. 

<img src="https://github.com/LinxiQIU/Attribute_Learning/blob/main/images/iterative.png" width="600" height="320">

```python
CUDA_VISIBLE_DEVICES=0,1 python train_attr_iteration.py --exp_name iterative --change adamw_wseg --with_seg True --epochs 200 --root /home/ies/dataset/dataset1000
```