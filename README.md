# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

This project implemented FCN-8s-VGG16 proposed in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038v2) using Tensorflow.

![sample][0]

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training and test images.

##### Run
Run the following command to run the project:
```
python main.py
```

This will train the model (it will check `./model/model.ckpt` to pick up the training, if no file it will train from scratch), save model as `model.ckpt` and frozen `.pb` file, and process all the test images in test folder.

**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### VGG16
When the project is run, the frozen `VGG16` model will be downloaded. The model can be found [here][1], too.

### Understanding VGG16

* Running the script `python vis_tfboard.py` will generate a folder `vgg16logdir` which you can exploit using `tensorboard --logdir="vgg16logdir"`. Go to the address designated using web browser to see the graph and Operation nodes. Look for `layer3_out`, `layer4_out` and `layer7_out`. You will understand why we want the outputs of these nodes. It's very clear that for the 3 dense layers, two of them are converted to the fully convolutional layers and the final layer is removed.

* Load the graph and list all the Operations and the outputs attached to them by running the script `python explore_vgg.py`. Doing this is more difficult for you to find which outputs of which layers you want to capture. Too many nodes in VGG16.

### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow.
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy.
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

[0]: ./runs/1533646882.4119525/um_000007.png
[1]: https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip
