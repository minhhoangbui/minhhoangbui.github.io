---
layout: post
title: Clustering and Retrieval
---

At the moment, we are working in a big project of personalized news distribution for 24h News. In this project, we assume that each reader of this page has their own taste, so we can't deliver the same content to them: we have to explore their favorite and then deliver news based on that observation. However, at the production scale, suppose we have millions of clients, the generation of millions of different content seems extreme expensive. So we have the idea of clustering our customers into N clusters, then generate N different contents for the cluster centroids. This idea seems plausible to us. In this blog, I want to tell you about the journey of finding the best clustering algorithm.

# I. What is retrieval and what is clustering

Giving an item, *retrieval* helps you to find a similar item to that by calculating the distances between the given item and every other items in the item space. To do this, the most important thing is to find an appropriate metric which helps you to compute the similarity between things.

<p align="center">
 <img src="/img/clustering-retrieval/retrieval.png" alt="" align="middle">
 <div align="center"> Retrieval</div>
</p>

On the other hand, clustering helps you to group the similar items in the space. Items with similar characteristics will be near each other and then we assume that they are in the same cluster. Obviously, every problems which is related to similarity is in need of a distance metrics. Each cluster will be represented by its centroids, which is in most cases the average point of the items in that cluster.

<p align="center">
 <img src="/img/clustering-retrieval/clustering.png" alt="" align="middle">
 <div align="center"> Clustering</div>
</p>

# II. Nearest Neighbor Search

Nearest Neighbor Search is the very symbolic technique of Retrieval method. In this approach, we compute the distance between the anchor and the other items in the space. For example, you are reading an article and you want to read the similar articles. Every articles will be represented by vector. Then we compute the distance, maybe with L2-distance so we can get the article with the minimum distance.

There are two types: query the most similar item (1-NN algorithm) and query the most k similar items (k-NN algorithm):

## 1-NN algorithm

The idea is pretty simple, just like what I said above

* Pseudo-code:

```{r, eval=FALSE, tidy=FALSE}
Initialize Dist2NN = $$\infty$$
For i = 1, 2, .., N:
    compute $$\delta$$ = distance($$x, x_i$$)
    if $$\delta$$ < Dist2NN:
        Dist2NN = $$\delta$$

return the item with distance Dist2NN
```

## k-NN algorithm

In this algorithm, we keep the record of most k similar items

```{r, eval=FALSE, tidy=FALSE}

Initialize Dist2kNN = sorted($$\delta_1, \delta_2, .., \delta_k$$)

for i = k+1, .., N:
    $$delta_i$$ = distance($$x, x_i$$)
    if $$\delta_i$$ < Dist2kNN[k]:
        insert $$\delta_i$$ into the sorted list Dist2kNN

return the items with the distance in Dist2kNN
```

The idea of the algorithm is really simple and intuitive, however, to guarantee the efficiency of the algorithms, there are 2 critical elements that need considering carefully: the vectorized representation of items and distance metrics.

### Vectorization

In case of text, there are many methods that could help us to vectorize the documents, like:

* Bag of Word (BoW): It is quite similar to the histogram in the visual domain. We count the occurrence of each word in the corpus, then make the vector with that. There is a major drawbacks with this idea: There are some words which appear in almost every document like *a, the, you, etc.*

* TF-IDF: It uses the same idea of word count, but TF-IDF penalize the score of words which appear in different documents in the corpus. Differently speaking, it favors the locally common words but disfavors the globally common words. TF stands for Term Frequency, IDF stands for Inverse Document Frequency. Details about this approach can be found easily from the Internet.

### Distance metrics

1. Scaled Euclidean Distance

$$ distance(x_i, x_q) = \sqrt{a_1(x_i[1] - x_q[1])^2 + .. + a_d(x_i[d] - x_q[d])^2}$$

$$a_1, .., a_d$$ are the weights for each feature. This feature selection is in fact not very popular in the real life since feature engineering is always hard for us. So the non-scaled version is more preferable:

$$ distance(x_i, x_q) = \sqrt{a_1(x_i[1] - x_q[1])^2 + .. + a_d(x_i[d] - x_q[d])^2}$$

2. Cosine similarity

$$ distance(x_i, x_q) = \frac{x_i^\mathsf{T} x_q}{\norm{x_i} \norm{x_q}}$$
