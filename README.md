# torch-reweighter

This project aims to develop a post-hoc correction for particle physics simulations. Simulations of lower accuracy can be produced with reduced computational cost. An additional correction step then reweights the degraded observables back to the nominal high accuracy ones. Overall, CPU processing time is reduced while acceleration hardware is utilised.

The correction is based on a NN classifier trained to discriminate two samples of data (degraded vs nominal) and the classification score will be used for multivariate reweighting of the degraded back to the nominal observables. 

The idea is based on [Approximating Likelihood Ratios with Calibrated Discriminative Classifiers](https://arxiv.org/abs/1506.02169).

The data are provided in the form of 3D images in HDF5 format. The current approach uses a 3D Convolution NN for images classification, similar to what is used within the computer vision domain. An initial model can be found in [models.py](https://github.com/ekourlit/torch-reweighter/blob/main/models.py).
