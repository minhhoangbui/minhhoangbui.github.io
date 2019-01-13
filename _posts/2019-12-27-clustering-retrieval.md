---
layout: post
title: Clustering and Retrieval
---

At the moment, we are working in a big project of personalized news distribution for 24h News. In this project, we assume that each reader of this page has their own taste, so we can't deliver the same content to them: we have to explore their favorite and then deliver news based on that observation. However, at the production scale, suppose we have millions of clients, the generation of millions of different content seems extreme expensive. So we have the idea of clustering our customers into N clusters, then generate N different contents for the cluster centroids. This idea seems plausible to us. In this blog, I want to tell you about the journey of finding the best clustering algorithm.

# I. What is retrieval and what is clustering?

Giving an item, *retrieval* helps you to find a similar item to that by calculating the distance between the given item and every other items in the item space. To do this, the most important thing is to find an appropriate metric which helps you to compute the similarity between things.


On the other hand, clustering helps you to group the similar item in the space. Items with similar characteristics will be near each other and then we assume that they are in the same cluster. Obviously, every problems which is related to similarity is in need of a distance metrics. Each cluster will be represented by its centroids, which is in most cases the average point of the items in that cluster