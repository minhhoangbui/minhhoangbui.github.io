---
layout: post
title: Randomization in Programming
---

Randomization is always an important techniques in programming and deep learning practiques in more specific. It sometimes plays the secondary role as solution initialization, the other times as a tool to solve the problems directly. In this blog, I will collect all of my knowledge about this sub-domain and present it to you.

# General idea

When we talk about randomization, we can't ignore the symbolic games of randomization: Gambling. Do you like gambling? Personally, I love the game!!! Under the mask of randomization and luck, it is actually a game of statistics and memorization. If you can really master these two techniques, no casino will be able to lighten your pocket and you are likely in their black list.
Back to the topic, gambling has two most famous capitals: Las Vegas and Monte Carlo. Some asians will argue about Macao, but honestly, it is not comparable to the other two in aspect of prestige. Why I mention this? Because two main branches of randomization in programming are named after the locations: Las Vegas approach and Monte Carlo approach. A fusion of the two techniques will be found easily in deep learning.

# Las Vegas approach

Las Vegas uses randomization as a way to initialize the parameter for the problems. After a while running the program, no matter what you initialize, it will return the solution if possible or report failure. Better initialization of course will help to find the answer more efficiently. In other words, the method doesn't gamble the solution, it gambles the resources and time to run the algorithm. An good important for this method is Quick Sort algorithm:


# Monte Carlo approach

Unlike Las Vegas, Monte Carlo usually use the randomization to solve the direct problems. It loosen the accuracy requirement. It doesn't have to find the exact solution as Las Vegas, but the more we use the resource, we could find the more finesse answer for the problems. We clearly can see the trade-off here: Spend more resources to find more exact solution or satisfy with the current one. The resource here is likely to understood as the number of sample. Surely, the Monte Carlo is more flexible for us, we always can find a solution at any times but the quality depends on the our effort, unlike the all-or-nothing fashion of Las Vegas. Moreover, different initialization will lead to different solutions even if they are at the same steps. However, it is not really important to us. After all, Monte Carlo is an uncompleted algorithm: we cannot finalize the execution of the programming. So it is obvious that different initialization will create a different in-process result. Maybe in some indefinite point, the results will converge to the unique one, it is not something we care.

An example for this method is how we compute the integral?

# Fusion of Las Vegas and Monte Carlo

It definitely exists in neural network. If neural network were a person, it was sure as hell he would carry double nationality: Americans and French (just kidding). In neural network, weight initialization will be carried by randomization. Many schemes for this step are proposed and the better scheme will lead to the solution more without much effort. But neural network cannot find the exact solution with the randomization, but we can believe that we put more effort in the training process like data collection (i.i.d samples) and training steps, we could come up with better model.


