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
<p align="center"> $$E_p = \Pr(\Upsilon(S_i) \ne \Upsilon(S_j) | S_i \Vdash_C S_j)$$ </p>

The false negative error rate of $$C$$ is given by:

<p align='center'> $$E_n = \Pr(\Upsilon(S_i) = \Upsilon(S_j) | S_i \nVdash_C S_j)$$ </p>

## 1. Spatial-Temporal Pattern Learning

Due to camera network topology, the time interval of pedestrian passing through different camera usually follow a 
pattern. These non visual information can boost the performance of the whole system considerably.  
Formally, the spatial-temporal can be defined as:
<p align='center'> $$ \Pr(\bigtriangleup_{ij}, c_i, c_j | \Upsilon(S_i) = \Upsilon(S_j)  with \bigtriangleup_{ij} = t_i - t_j $$</p>  

This indicates the probability distribution of the time interval $$\bigtriangleup_{ij}$$ and camera ID $$(c_i, c_j)$$ of 
any pair of $$S_i, S_j$$ given that they belong to the same ID. However, to learn this knowledge from the unsupervised 
dataset is impossible since we don't know about their ID. One possible option is to approximate the above probability with 
the help of transfer learning. Applying the classifier $$C$$ to every pair of images can estimate the statistics 
$$\Pr(\bigtriangleup_{ij}, c_i, c_j | S_i \Vdash_C S_j)$$ and $$\Pr(\bigtriangleup_{ij}, c_i, c_j | S_i \nVdash_C S_j)$$. 
The detail of computing these probabilities can be found in the paper. From the above number, we can infer that:  

<p align='center'> $$ \Pr(\bigtriangleup_{ij}, c_i, c_j | \Upsilon(S_i) = \Upsilon(S_j) = (1 - E_p - E_n)^{-1} ((1 - E_n) 
* \Pr(\bigtriangleup_{ij}, c_i, c_j | S_i \Vdash_C S_j) - E_p * \Pr(\bigtriangleup_{ij}, c_i, c_j | S_i \nVdash_C S_j)$$ </p>

## 2. Bayesian Fusion model

The fusion model is based on the conditional probability:  
<p aligh='center'> $$\Pr(\Upsilon(S_i) = \Upsilon(S_j)| v_i, v_j, c_i, c_j, \bigtriangleup_{ij}) $$ </p>

$$v_i, v_j$$ is the feature vector of $$S_i, S_j$$

According to Bayesian rule:

<p aligh='center'> $$\Pr(\Upsilon(S_i) = \Upsilon(S_j)| v_i, v_j, c_i, c_j, \bigtriangleup_{ij}) = \frac
{\Pr(\Upsilon(S_i) = \Upsilon(S_j)| v_i, v_j) * \Pr(c_i, c_j, \bigtriangleup_{ij}|\Upsilon(S_i) = \Upsilon(S_j))}
{\Pr(c_i, c_j, \bigtriangleup_{ij})} $$</p>  

The first term in the denominator can be deduced from the matching score from the traditional model. The numerator can 
be computed through counting. More detail can be found in the original paper.  

## 3. Incremental Optimisation by Learning-to-rank,

Given visual classifier $$C$$ and fusion classifier $$F$$, the authors have proven that $$F$$ is an upgraded version of
$$C$$ with better performance. So it emerges an idea that if we could use the ranking of $$F$$ for the unlabeled data to 
train $$C$$, subsequently, the performance of $$F$$ will get better. In my opinions, it is inspired from the duality 
principle in optimization. I won't discuss in more detail due to the fact that their implementation at the moment doesn't 
fit our system, however, it is an idea that is worth considering in the future.

# II. Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification

- [Paper](https://arxiv.org/pdf/1904.01990.pdf)
- [Source code](https://github.com/zhunzhong07/ECN)

In this paper, they propose several ideas to integrate the unlabeled data, nevertheless, I find most of the ideas not 
persuasive and infeasible to mobile device. But the idea of using GAN to generate cropped images from other camera is 
interesting somehow. It not only help to augment the data considerably but also save us from labeling data.

This technique makes use of a variation of GAN which is named CycleGAN. Given two datasets $${x_i}, {y_i}$$ from two 
different domains $$A$$ and $$B$$. The goal of this architecture is to learn the mapping function $$G: A \leftarrow B$$ which 
satisfies $$G(A)$$ is indistinguishable from distribution $$B$$. CycleGAN includes two mapping function $$G: A \leftarrow 
B, F: B \leftarrow A$$. Two conditions that the mapping function must satisfy:  

- $$G(A) \sim B, F(B) \sim A $$
- $$G(F(B)) \sim B, F(G(A)) \sim A$$

In CycleGAN, there are two types of loss functions:

#### 1. Adversarial Loss  

For the mapping function and its discriminator $$(G, D_Y)$$, we infer the GAN Loss as following:

<p align='center'> $$\mathcal{L}_{GAN}(G, D_Y, X, Y) = \mathbb{E}_{y \sim Y} log(D_Y (y)) + \mathbb{E}_{x \sim X} log(1 - 
D_Y(G(x)))$$ </p>  

This goes the same for $$(F, D_X)$$

#### 2. Cycle Consistency Loss  

To guarantee the second condition, we employ an additional loss:

<p align='center'> $$\mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x \sim X}\|F(G(x)) - x \| + \mathbb{E}_{y \sim Y}
\| G(F(y)) - y \| $$ </p>
