---
title: "Why pure vision isn't scaling"
date: 2026-05-21 09:00:00 -0400
categories:
  - blog
---

## Contents
{: .no_toc }

* Auto generated table of contents
{:toc}

## What I Mean by Isn't Scaling

It was pointed out to me that this claim could be taken literally. It shouldn't be. If you make the model bigger, some benchmark numbers will get better. Rather, it's better read as "why do we not have an objective that when we scale, sufficiently solves all of our demands for vision"?


## The Drivers of a Scalable Objective

For the purpose of this blog, scaling is:

> Bigger data + bigger model -> learning more generalisable abstractions from data, and therefore getting a better model.

The primary drivers of scalability seem to be a notion of **objective signal-to-noise**, as well as the quality and quantity of the objective's **total accessible information**. 

Objective SNR:
* Every gradient step on the objective needs to have sufficient signal to bias the model toward learning abstractions rather than noisy or spurious explanations. 

Quantity of information:
* There should also be enough information in the objective for that objective signal-to-noise ratio to be useful at scale. 

Quality of information: 
* This information should be semantically broad/rich enough that the model has to learn cohesive abstractions (that can generalize) rather than fragmented features. E.g., features only useful for classifying some set of animals aren't great abstractions, and so narrow supervised training (e.g., ImageNet classification) doesn't produce great abstractions. 

## Objective signal-to-noise is not data signal-to-noise

For a given data signal-to-noise ratio, you can have varying amounts of objective signal-to-noise depending on your objective design.

Let's consider the semantically broad task of image generation, where we have both quantity and quality under our definition. The diffusion objective's denoising factorization has higher objective signal-to-noise than naive one-shot prediction tasks like the VAE. We observe that diffusion objectives are more scalable than naive VAE objectives for pixel-space generation. 

## Why Text Scales

For text, two things seem especially important:

1. Data signal-to-noise is high, so the simple supervision of token prediction has little noise. Objective SNR falls out of the data SNR being high.

2. We have a lot of data that we directly predict, so the quantity is good. This data has very broad semantic coverage, so the quality is good. 


## Why Vision Doesn't Scale

For images, pure pixel prediction is too noisy, which hurts objective SNR. We attempt to address this with latent self-supervised learning, which broadly performs the trade of increasing objective signal-to-noise by suppressing accessible information.

Suppressing the accessible information has two main problems:
1. We obviously decrease the quantity of information.
2. We also decrease the breadth of semantic coverage that the objective demands, as the suppressed information is biased. We suppress regions of the abstraction hierarchy from the learning signal (e.g., higher spatial frequency details), so it's hard to learn factors predictive of those regions in the final representation. 

**It is hard to create a scalable objective that ignores any of our three main factors.**

The search for a vision scaling law probably requires an objective that improves objective signal-to-noise without permanently suppressing accessible information. I'm unsure if purely latent methods can ever provide this. 

## Footnotes
When I've say "information", be sure to understand I'm referring more to some notion of information **accessible** to your model, objective, and compute bounds. 