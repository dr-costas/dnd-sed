# Sound event detection with depthwise separable and dilated convolutions

----

### Welcome to the repository of DnD-SED method. 

This is the repository for the method presented in the paper 
"Sound Event Detection with Depthwise Separable and Dilated Convolutions", by 
[K. Drossos](#https://tutcris.tut.fi/portal/en/persons/konstantinos-drosos(b1070370-5156-4280-b354-6291618bb965).html), 
[S. I. Mimilakis](#https://www.idmt.fraunhofer.de/en/institute/doctorands/mimilakis.html), 
[S. Gharib](#), 
[Y. Li](#), 
and [T. Virtanen](#https://tutcris.tut.fi/portal/en/persons/tuomas-virtanen(210e58bb-c224-40a9-bf6c-5b786297e841).html).

Our code is based on [PyTorch framework](#https://pytorch.org/) 
and we use the publicly available dataset 
[TUT-SED Synthetic 2016](#http://www.cs.tut.fi/sgn/arg/taslp2017-crnn-sed/tut-sed-synthetic-2016). 

Our paper is submitted for review to the [IEEE World Congress on Computational 
Intelligence/International Joint Conference on Neural Networks 
(WCCI/IJCNN)](#https://wcci2020.org/).  

You can find an online version of our paper at arXiv: __url to be announced__

**If you use our method, please cite our paper.**  

----

## Table of Contents
1. [Method introduction](#method-introduction)
2. [System set-up](#system-set-up)
3. [Conducting the experiments](#conducting-the-experiments)

----

## Method introduction

Methods for sound event detection (SED) are usually based on a composition
of three functions; a feature extractor, an identifier of long temporal context, and a
classifier. State-of-the-art SED methods use typical 2D convolutional neural networks (CNNs)
as the feature extractor and an RNN for identifying long temporal context (a simple 
affine transform with a non-linearity is utilized as a classifier). This set-up can 
yield a considerable amount of parameters, amounting up to couple of millions (e.g. 4M)
Additionally, the utilization of an RNN impedes the training process and the parallelization
of the method.  

With our DnD-SED method we propose the replacement of the typical 2D CNNs used as a 
feature extractor with depthwise separable convolutions, and the replacement of the
RNN with dilated convolutions. We compare our method with the widely-used CRNN method,
using the publicly available TUT-SED Synthetic 2016 dataset. We conduct a series of 
10 experiments and we report mean values of time needed for one training epoch, F1 score,
error rate, and amount of parameters.   

We achieve a considerable decrease at the computational complexity and a simultaneous
increase on the SED performance. Specifically, we achieve a reduction of the amount of 
parameters and the mean time needed for one training epoch (reduction of 85% and 72% 
respectively). Also, we achieve an increase of the mean F1 score by 4/6% and a reduction
of the mean error rate by 3.8%. 

You can find more information in our paper!

----

## System set-up

To run and use our method (or simply repeat the experiments), you need to set-up
the code and use the specific dataset. We provide you the full code used for the
method, but you will have to get the audio files and extract the features.   

### Code set-up

To set-up the code, you will need 

### Data set-up

----

## Conducting the experiments
