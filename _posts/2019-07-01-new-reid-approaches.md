---
layout: post
title: Integrating unsupervised learning to ReID problem
---

At the moment, we are working on a ReID system in retail market. It aims to recognize people through a large system of 
cameras, by that we can discover the buying patterns of customer and improve client's experience.  
From my point of view, the most important element in a ReID system is the vectorization which transform a cropped image 
of a specific person into a distinguished vector. Most vectorization systems are based on closed dataset like Market-1501
, CUHK, etc. which are both not general and expensive to obtain. The new trend is to use both supervised dataset and 
unsupervised dataset in order to achieve better performance in real-life applications.


# I. Unsupervised Cross-dataset Person Re-Identification by Transfer Learning of Spatial-Temporal Patterns

- [Paper](https://arxiv.org/pdf/1803.07293)
- [Source code](https://github.com/ahangchen/TFusion)

This paper includes the following contributions:
- Transfer model trained from fully-labeled dataset to unlabeled dataset, hence, to learn spatial-spatial relationship
between images from different cameras
- Incorporate the spatial-temporal knowledge into the visual features to achieve high performance in ReID in unlabeled
dataset
- Proposing a learning-to-rank in which the fusion model in this iteration teach the weaker classifier, thereby improve 
fusion model in the aftermath.

## 0. Preliminaries

Each cropped image is denoted as $$S_i$$, the capture time and the capture camera are $$t_i, c_i$$. The ID of the 
pedestrian in $$S_i$$ is denoted as $$\Upsilon(S_i)$$.  
The traditional strategy of Person ReID is to train a supervised classifier $$C$$ based the visual features to judge whether 
two given cropped frame belong to the same ID or not. If it is, we denote as $$S_i \Vdash_C S_j$$. Otherwise, 
$$S_i \nVdash_C S_j$$.

The false positive error rate of $$C$$ is given by:  
$$E_p = Pr(\Upsilon(S_i) \ne \Upsilon(S_j) | S_i \Vdash_C S_j)

The false negative error rate of $$C$$ is given by:

$$E_p = Pr(\Upsilon(S_i) = \Upsilon(S_j) | S_i \nVdash_C S_j)

## 1. Spatial-Temporal Pattern Learning