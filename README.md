# cpp_xoczy_ai

This repository contains examples of simple neural network implementations in C++ designed for image recognition tasks. The program is capable of learning from a set of training images, which are stored in text files, and can subsequently recognize patterns in a separate test image file.

The training images are represented as binary data, where each pixel is denoted by either a 0 (white) or 1 (black), forming a grid structure. The network learns to associate the patterns present in these training images with expected outputs, which indicate the presence or absence of specific features.

Once trained, the network can analyze a test image (also stored as a binary text file) to determine whether it contains the learned patterns. The program includes a straightforward implementation of forward propagation for making predictions and backpropagation for updating the weights during the training phase.

This setup allows users to experiment with various shapes and patterns in the training images, enabling the network to generalize and potentially recognize similar patterns in unseen images. By modifying the training images, users can assess the network's ability to adapt and identify various shapes beyond those it was explicitly trained on.
