---
layout: post
title: Apache Hadoop Architecture
---

Recently, I've taken part in a online course in order to augment my salary. It is about Big Data and Apache's tools in this area. So I decide to write a series about this topic. This series will act as a summary for my learning and also help to enrich the content of this site. In this course, they teach us about doing machine learning at scale and the way deploying machine learning model into production environment. That sounds great, right? So let's begin.

# I. What is Big Data

Based on the definition of [Apache Hadoop Tutorial](https://www.youtube.com/watch?v=mafw2-CVYnA):

> Big data is the term for a collection of data sets so large and complex that it becomes difficult to process using on-hands database management tools or traditional data processing applications.

Traditionally, the type of data we can imagine is the Excel sheet with a load of number and we can use Excel or some similar application to process and exploit useful information from it. Of course, it is not easy for an amateur to deal with the formulas in Excel but at least it has its own format for us to consider.

Nowadays, everything you create while surfing the internet can be consider as data and since it is enormous, so we name it Big Data. That Big Data provides a shape of who you are, where you are, what you want, etc. to the companies so that they could use it for a better market strategy. A picture could tell us where you live, click stream in a e-commerce site could help us to predict what you want in the future, etc. Sometimes, with this data, they could understand you more than you do. I strongly recommend this [book](https://www.amazon.com/Everybody-Lies-Internet-About-Really/dp/0062390856), it will help you to realize the power of Big Data in this age.

Back to the topic, there are five properties of Big Data that you should notice:

- Volume: The amount of data is so huge that we could not store it and process it efficiently in the traditional system.

- Variety: The types of data now are so various, from text to music, from video to even click-stream. Furthermore, for each type of data, we also have many kinds of format to store them. For example, mp3 and wav in music or .doc and .json for text.

- Velocity: Data is generated at a massive rate. In the past, not many people could possess a digital device to create data, but now, there are everywhere. So traditional system cannot keep up with this paces of data generation.

- Value: Since the data is everything and everywhere, we have to find a way to exploit its value or else, it is just a piece of crap. Traditional systems cannot deal with this because of their processing power limit.

- Veracity: Data always contains noises, not everything they tell us is true. So how to deal with this inconsistency is also a big problems to solve
