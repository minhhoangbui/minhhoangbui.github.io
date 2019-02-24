---
layout: post
title: Clustering and Retrieval (Part 2)
---

In this blog, I will dive deeper into the techniques of clustering. Clustering or segmentation has wide applications, particularly in RecSys, when you can have the insight about users preferences given their activities. For examples, assuming that we have a list of articles that user A reads everyday, we can extract his favorite topics and then give him what he likes.

# I. Clustering properties

Clustering, in general, like retrieval, is an unsupervised task. We study the features of the input and then decide which group the input belongs to. Maybe in some cases, the clustering may seem similar to the classification, but the classification has label for each input whereas the clustering doesn't.

<p align="center">
 <img src="/_image/clustering-retrieval/retrieval.png" alt="" align="middle">
 <div align="center"> Retrieval</div>
</p>

So what is a cluster and what characterizes it? A cluster in space is a group of similar points which stay near each other. Each cluster is defined by its centroid and its shape. An observation $$x_i$$ is assigned to cluster $$C_j$$ if the score between $$x_i$$ and $$C_j$$ is the smallest in comparison to other clusters

There are many topologies which challenges the data scientist to cluster:

<p align="center">
 <img src="/_image/clustering-retrieval/retrieval.png" alt="" align="middle">
 <div align="center"> Retrieval</div>
</p>

# II. K-Means Clustering

K-means is a symbolic approach in Clustering. In K-Means, the score is the distance between the observation and the centroids of the clusters (the smaller the better). This method helps us to divide our elements set into k clusters

In K-Means, there are 4 steps:

1. Initialize the cluster centroids(They are preferably k points from the space)

2. Assign the observation to the closest cluster using the distance between the observation and the centroids

$$ z_i = argmin_j ||\mu_j - x_i||^2_2

3. Update the coordinates of the clusters using the mean of every points assigned to that cluster

$$ \mu_j = \frac{1}{n_j} \sum_{i: z_i=j} x_i

4. Repeat step 2 & 3 until convergence

In fact, the above process is inspired from _E-M Optimization_. It includes 2 step: Expectation and Maximization until convergence. It has another name: _alternating minimization_

K-Mean guarantees that we could reach the local optimum for the problems. It also means that the result of the algorithm strongly depends on the initialization of the centroids. Furthermore, it also depends on the hyperparameter $$k$$. To fine-tune this parameter, there is a techniques called _elbow method_. You can get more detail about it from the Internet.

As you can see, initialization plays an important role in K-Mean, then if we could somehow optimize it, K-Means will be more effective. This is why they invent _kmean++_

## K-Means ++

The intuition of the method is to choose the centroids which are far from each other so that the boundary between the cluster will be more well-defined.

Algorithm:

1. Choose the first centroid randomly from the data points

2. Compute the distance between every data point and the chosen centroid

3. Choose the next centroid from the data points with the probability which is proportional to the distance to other centroids. We hope the next centroid will be far from the chosen centroids

4. Keep doing the step 2 & 3 until we have $$k$$ centroids.

5. Do the *E-M Optimization* like the normal K-Means

