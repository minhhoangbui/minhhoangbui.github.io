---
layout: post
title: SSD vs YOLO
author: hoangbm
---

To be honest, I'm truly fed up with revising knowledge before an interview, especially object detection algorithms like SSD and YOLO. Every time I prepare for an interview in computer vision, I read about these two architectures and I can't tell the exact differences between them. So today, I decide to put an end to this, so that I can move on with a smile :).

Obviously, you may tell that SSD is relatively obsolete nowadays. At the time I type this blog, YOLO has been so dominant in the world of object detection world. YOLO reaches [its 7th version](https://github.com/WongKinYiu/yolov7) even though most of its upgrades are about things like data augmentation, preprocessing, etc. It seems to me that the core architecture has remained unchanged since YOLOv4, correct me if I'm wrong. SSD and even two stage detector like Mask-RCNN seems to fall behind YOLO for too long. So why the heck spending for this blog when YOLO is unarguably winner in this area. Well, I write this blog, as I said, to draw a clear boundary between SSD and YOLO and put my mind about this at ease. Furthermore, I think it is for my future interview, people in the industry still like to ask about these topics during interview. My next topic will be about Batch Norm.

Since there are many versions of YOLO, I will choose YOLOv1 and YOLOv3 to talk about. And I focus only on 2 aspects: architecture input/output and loss function.

## I. Architecture Ouput

You may wonder why I priotize this the most (indicated by the section order). Well, as a machine learning engineer, I care less and less to training paradigm details. I only care about how the architecture tackles with the problem that I have at hand and learning about input/output is the perfect way for me to grasp the idea of what the heck this model is doing.

Both YOLO and SSD take a 4D tensor (with batch dimension) during training and 3D tensor during inference. For the sake of simplicity, I assume that the model take a tensor of (C * H * W) as input. That's the easy part; almost every computer vision model does that. What about the output?

SSD embraces the idea of multiscale detection, hence, at each of last layers, it produces a set of prediction. Given an image of input `1 x 300 x 300 x 3`, this creates a set of outputs:

```
Say for example, at Conv4_3, the output is 38×38×4×(c+4). 
In terms of number of bounding boxes, there are 38×38×4 = 5776 bounding boxes.
Similarly for other conv layers:
Conv7: 19×19×6 = 2166 boxes (6 boxes for each location)
Conv8_2: 10×10×6 = 600 boxes (6 boxes for each location)
Conv9_2: 5×5×6 = 150 boxes (6 boxes for each location)
Conv10_2: 3×3×4 = 36 boxes (4 boxes for each location)
Conv11_2: 1×1×4 = 4 boxes (4 boxes for each location)
```

<p align="center">
     <img src="/images/object_detection/ssd.png" alt="" align="middle">
     <div align="center">
        How SSD produces its output. </div>
</p>

The first version of YOLO don't give a damn about multiscale. The last Conv layer produces a fixed set of outputs. there are `7×7` locations at the end with 2 bounding boxes for each location. YOLO only got 7×7×2 = 98. This is a reason why some first versions of YOLO don't work stably, especially when it comes to video stream. It can detect well object in one frame and fails at the next with almost no difference between those two. Their potential candidates are too few in comparison with SSD. The huge amount of anchor boxes in SSD contribute a lot to its stability when working with video.

<p align="center">
     <img src="/images/object_detection/yolo_v1.png" alt="" align="middle">
     <div align="center">
        How YOLOv3 produces its output.
    </div>
</p>

That's the reason why YOLOv3 implements multiscale detection. With an image of input `1 x 416 x 416 x3`,
this creates a output in form of a list of 3 `[(1 x 13 x 13 x 125), (1 x 26 x 26 x 125), (1 x 52 x 52 x 125)]`.

Why 125 here?

Suppose for each grid cell we predict 5 bounding boxes, each bounding box has 4 coordinate values and are 21 classes, then `(20 + 5) * 5 = 125`.

Another thing is that while SSD considers background as a class, YOLO gives another meaning to this number, `objectness score`, which indicates the probability there is a object in this grid cell. It's more intuitive, in my opinion. Considering `background` as a class will lead to imbalance between this class and the others and this can't be remedied effectively by hard negative sampling. In short, in YOLO, it is `20 + 4 + 1`, while in SSD it's `21 + 4`.

<p align="center">
     <img src="/images/object_detection/yolo_v3.png" alt="" align="middle">
     <div align="center">
        How YOLOv1 produces its output.
    </div>
</p>

Basically speaking, the format of outputs is relatively the same between. There are some mismatches in the number of anchor boxes, actual meaning of coordinates value and objectness score. But I believe it's safe to say that it is similar.

## II. Loss function

I believe the second important factors of machine learning algorithm is its loss function. While input/output is the core of inference (I think), loss function is the heart of training process. In fact, understanding loss function and how it associates the output to the label is the only thing I deem important when training a model.

Up until now, you can see that when we compute the forward-pass, we keep mentioning the number of anchor boxes per cell to calculate the total number value each cell must output. However, other than that, we don't use them at all, we don't know where they are, their coordinates. So when do we use them?

The answer is we use it during loss computation and inference. Basically, object detector doesn't predict the bounding boxes directly like most people assume, instead it predicts the likelihood the corresponding anchor box contains object or not and its distance to the true bounding box. That's why we only need the anchor box's information when computing loss and during inference, remember `anchor box + prediction = bounding box`.

From the annotation, SSD and YOLO build its ground-truth label. The label tends to have 5 values: 4 coordiates and class info. They also share the idea of matching anchor-boxes to ground-truth labels. Roughly speaking, the anchor boxes are divided into 2 groups: positive and negative. Positive group includes boxes that has IoU with groundtruth bigger than a specified threshold and vice versa. Each anchor box will associate with only one ground-truth box or nothing at all (background). And the model will predict the `correction` of coordinates and the class label for each anchor box. Why I call `correction`? This is the key idea (I think I should put it in input/output?). The model don't predict the coordinates directly but the differences between this anchor box and its corresponding gt box.

Furthermore, as you can see, YOLO and SSD interpret objectness score differently, hence, the loss function will be different too.

To be more specific, here is SSD loss function:

$$ L(x, c, l, g) = \frac{1}{N}(L_{conf}(x, c) +\alpha L_{loc}(x, l, g))$$

In short, SSD loss function is formed by 2 sub losses: location loss and classification loss. While the first one is to form the location correction, the second one is to predict the class of this box. For the first loss, the authors use L1 regression to get the optimal value and for the second one, they simply use cross-entropy with (N+1) classes. Nothing fancy here.

About YOLO's loss function:

$$ \lambda_{coord}\sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x_i})^2 + (y_i - \hat{y_i})^2] + \\
\lambda_{coord}\sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(\sqrt{w_i} - \sqrt{\hat{w_i}})^2 + (\sqrt{h_i} - \sqrt{\hat{h_i}})^2] + \\
\sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C_i}^2) + \lambda_{noobj}\sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C_i}^2) + \\
\sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p_i(c)})^2
$$

Basically, there are 3 components: Location loss, classification loss and objectness classifcation. The first two terms indicate L2 Regression to find the wanted values. The next two terms will help to predict whether which anchor box contains object, tt's also L2. And the last one is cross-entropy loss, no brainer.

As you can see, since the number of anchor boxes in SSD is much more huge than in YOLO, the problem of imbalance between positive and negative group is much severe. SSD tackle with this problem with a complexed hard-mining sampling while YOLO only uses different weight for each group in order to balance the influence of each group.
I won't (or will I?) discuss the detail of negative hard sampling in this blog, but basically, instead of using all the samples in the negative group, we only use a subset which has quite a large IoU with gt box to compute the loss.

## III. Conclusion

That's basically so. In my opinion, these points are the main differences between SSD and YOLO. As it turns out, these two architectures are quite alike. I guess that's the reason why SSD remains the same for the last 7 years while YOLO progresses massively thoughout the years. YOLO is more optimized by design (fewer anchor boxes, the existence of `objectness score`, .etc) and it's quite easy for YOLO to adopt SSD's amazing features like multi-scale detection. YOLO supporters are quick to equip this architecture with these abilities. And eventually, these two original architectures converge into later version of YOLO.
Furthermore, they also try to leverage other techniques like data augmentation to boost the performance of YOLO.

Other than that, I haven't seen any improvement of architecture for the last couple of years. Let's see any progresses to be made in this area in the future. I've heard about `Nano-Dets` and the use of `Transformer` but it's for another day.
