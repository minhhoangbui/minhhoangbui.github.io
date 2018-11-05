---
layout: post
title: Randomization in Programming
---

Randomization is always an important techniques in programming and deep learning pratiques in more specific. It sometimes plays the secondary role as solution initialization, the other times as a tool to solve the problems directly. In this blog, I will collect all of my knowledge about this sub-domain and present it to you.

# General idea

When we talk about randomization, we can't ignore the symbolic games of randomization: Gambling. Do you like gambling? Personally, I love the game!!! Under the mask of randomization and luck, it is actually a game of statistics and memorization. If you can really master these two techniques, no casino will be able to lighten your pocket and you are likely in their black list.

Back to the topic, gambling has two most famous capitals: Las Vegas and Monte Carlo. Some asians will argue about Macao, but honestly, it is not comparable to the other two in aspect of prestige. Why I mention this? Because two main branches of randomization in programming are named after the locations: Las Vegas approach and Monte Carlo approach. A fusion of the two techniques will be found easily in deep learning.

# Las Vegas approach

<p align="center">
 <img src="/img/randomization/gn-gift_guide_variable_c.jpg" alt="" align="middle">
 <div align="center"> The lux of Las Vegas</div>
</p>

Las Vegas uses randomization as a way to initialize the parameters for the problems. After a while running the program, no matter what you initialize, it will return the solution if possible or report failure. Better initialization of course will help to find the answer more efficiently. In other words, the method doesn't gamble the solution, it gambles the resources and time to run the algorithm. An good important for this method is Quick Sort algorithm:

- We pick randomly a pivot from the array

- Rearrange the array so that all the elements with values lower than pivot value come before the pivot and vice-versa. After this partitioning, the pivot will be at the end.

- Recursively repeat the above step to separate the array into lower sub-array in the left side and greater sub-array in the right side.

```py
    def quick_sort(a_list):

        def _quick_sort(a_list, first, last):
            if first <  last:
                split_point = partition(a_list, first, last)

                _quick_sort(a_list, first, split_point - 1)
                _quick_sort(a_list, split_point + 1, last)

        def partition(a_list, first, last):
            pivot_value = a_list[first]
            left = first + 1
            right = last
            done = False

            while not done:
                while left <= right and a_list[left] <= pivot_value:
                    left += 1
                while a_list[right] >= pivot_value and left <= right:
                    right -= 1
                if left > right:
                    done = True
                else:
                    temp = a_list[left]
                    a_list[left] = a_list[right]
                    a_list[right] = temp

            temp = a_list[right]
            a_list[right] = a_list[first]
            a_list[first] = temp
            return right

        _quick_sort(a_list, 0, len(a_list) - 1)
```

# Monte Carlo approach

<p align="center">
 <img src="/img/randomization/16980.jpg" alt="" align="middle">
 <div align="center"> The charm of Monte Carlo</div>
</p>

Unlike Las Vegas, Monte Carlo usually use the randomization to solve the direct problems. It loosens the accuracy requirement. It doesn't have to find the exact solution as Las Vegas, but the more we use the resources, we could find the more finesse answer for the problems. We clearly can see the trade-off here: Spend more resources to find more exact solution or satisfy with the current one. The resources here are likely to understood as the number of samples and training time.

Surely, the Monte Carlo is more flexible for us, we always can find a solution at any times but the quality depends on the our effort, unlike the all-or-nothing fashion of Las Vegas. Moreover, different initialization will lead to different solutions even if they are at the same steps. However, it is not really important to us. After all, Monte Carlo is an uncompleted algorithm: we cannot finalize the execution of the programming. So it is obvious that different initialization will create a different in-process result. Maybe in some indefinite point, the results will converge to the unique one, it is not something we care.

An example for this method is how we compute the area of a circle of radius $$R$$ using the integral. We can brainstorm this formula:

$$ S = -4 \int_{\pi/2}^{0} R^2 \sin^2\alpha d\alpha$$

I can transform into a python code like this:

```py
    import math

    def compute_circle_area(R, n_samples):
        alpha = [math.pi / 2 / n_samples * i for i in range(n)]
        sin2_alpha = [math.sin(a) ** 2 for a in alpha]
        d_alpha = math.pi / 2 / n_samples
        return 4 * R **2 *d_alpha * sum(sin2_alpha)
```

For n_samples = 10, $$S= 282.743338823$$, pretty bad approximation.
For n_samples = 1000, $$S= 313.845106094$$, much better
For n_samples = 100000, $$S= 314.156123766$$, it is good enough for most applications.

In Monte Carlo, the degree of sampling does matter, unlike Las Vegas, no matter how much you sample the initial point, it will return the same solution.

# Fusion of Las Vegas and Monte Carlo

It definitely exists in neural network. If neural network were a person, it was sure as hell he would carry double nationality: Americans and French (just kidding). In neural network, weight initialization will be carried by randomization. Many schemes for this step are proposed and the better scheme will lead to the solution more without much effort. Good random initialization scheme will lead the model to a well-defined region where we could converge the result much faster.

Neural network cannot find the exact solution with the randomization, but we can believe that we put more effort in the training process like data collection (i.i.d samples) and training steps, we could come up with better model.

Since initialization plays a really important to get a good local optima, TensorFlow provides many function to initialize weights: *tf.contrib.layers.xavier_initializer*, *tf.initializers.he_normal*, *tf.initializers.truncated_normal*

# Conclusion

In this blog, I only give a very general introduction about randomization. In the upcoming, I might give a more specific intuition about sampling in machine learning. Hope it will help you gain more understanding about this mysterious area.
