---
title: "Are all scalable objectives alike? "
date: 2026-08-6 09:00:00 -0400
permalink: /blog/filling-the-hierarchy-part-1/
categories:
  - blog
---

Tolstoy begins *Anna Karenina* with the claim that all happy families are alike. I propose an answer to the analogous question for learning objectives: are all scalable objectives somehow alike?
{: .notice--primary .intro-notice}

## Contents
{: .no_toc }

* Auto generated table of contents
{:toc}

## Introduction

The goal of this post is to present the **hierarchy filling** intuition for how I'm currently thinking about representation learning. 

Here, I aim to provide a language which allows us to consider why, across different domains, some objectives scale while others don't. The current lack of a unified framework for thinking about this problem creates friction when attempting to apply core ideas across domains. 

This post will provide examples for applying this **hierarchy filling** intuition to LLMs and existing SSL objectives (DINO and MAE). 

There was a previous version of this post titled "Why pure vision isn't scaling well enough" that I wrote very quickly. I really dislike it and think it was poorly thought out. This is an updated version. 

An accompanying SSL paper which proposes an objective and architecture will be out soon. 

## Good (scalable) learning **fills the abstraction hierarchy**. 


<div class="notice--primary intro-notice hypothesis-notice">
  <p><strong>Hypothesis:</strong> All scalable objectives fill the abstraction hierarchy as you scale.</p>
  <img src="/images/posts/scaling-law-for-pure-vision/full-hierarchy.png" alt="Full abstraction hierarchy">
</div>


We want scaling to better compose **both** higher level semantics, and the lower level abstractions which support them. An objective that biases towards filling the learned abstraction hierarchy as you scale is a scalable objective. 

Why do we want both this height (builds up) and width (filled at each level) as we scale?
* A lack of higher level semantics is generally less useful for modeling, even if lower level abstractions are good. 
* A lack of good lower level abstractions, other than being bad for tasks that need them, inhibits learning robust higher level semantics. Intuitively, composing together better lower level abstractions gives better higher level semantics. 

Mirroring this, scalability seems to break down when we are missing one of these factors:
* Case 1: Scaling could learn better lower level abstractions, but doesn't sufficiently bias towards composition that builds the hierarchy up towards higher level semantics. 

<div class="case-diagram">
  <img src="/images/posts/scaling-law-for-pure-vision/weak-top.png" alt="Weak top of the abstraction hierarchy">
</div>

* Case 2: Scaling doesn't learn more/better lower level abstractions, which inhibits the quality and quantity of higher level semantics you can learn. 

<div class="case-diagram">
  <img src="/images/posts/scaling-law-for-pure-vision/weak-bottom.png" alt="Weak bottom of the abstraction hierarchy">
</div>

## Why does next token prediction (NTP) scale well? 

1. **NTP builds up:** Predicting a high SNR + higher semantic target is a bias towards learning higher level semantics. The signal-to-noise of text is high so predicting text is a high SNR target, and we can pick the text so it contains higher level semantics. 

    For some high level concept C, its text tokens are often a strong signal for concept C, and so predicting concept C's tokens is predicting a high SNR + higher semantic target. 

2. **NTP fills each level:** Principally, assuming our data has broad coverage, to fill (learn good abstractions for) a given level, the objective should provide signal for how said level is composed, and how it composes into higher levels. 

    Learning to compose increasing levels of abstraction emergent from the data provides this signal for all levels (I've previously referred to this as supervising semantic construction). For a given abstraction at a given level: 

    - **Predicting it** signals how that level is composed, and how its lower levels are used in composition.
    - **Predicting using it** learns how that level composes into higher levels.
    
    In text, earlier concepts (say concepts A and B) compose into later concepts (say just concept C). To predict some tokens associated with the later concept C, the model learns those earlier concepts A and B. You can recurse this idea starting from the lowest level (the token). So, next token prediction is an effective proxy for learning to compose increasing levels of abstraction. 

NTP avoids both failure cases, and does indeed scale well. 

## Why does vision not scale well? 

Visual learning currently has two camps, one which predicts the data as the target, and one which bootstraps a latent target. Neither camp has good enough scaling behavior that we find in LLMs. More precisely, naively scaling SSL currently produces less clean and less predictable gains than scaling language models. More ambitiously, why is it that with existing methods and compute, can we not learn mathematics like language models do, just from visual data? To concretise our discussion, we'll consider the MAE and DINO. 

**The MAE objective biases towards a <span class="case-term" tabindex="0" data-tooltip="Case 1: Scaling could learn better lower level abstractions, but doesn't sufficiently bias towards composition that builds the hierarchy up towards higher level semantics.">Case 1 failure</span>:**

<div class="case-diagram">
  <img src="/images/posts/scaling-law-for-pure-vision/weak-top.png" alt="Weak top of the abstraction hierarchy">
</div>

The MAE predicts masked out pixels with patch representations, which isn't a good strong bias towards learning higher level semantics. Scaling does not change this. 

Empirically, we can observe that the MAE performs worse on global tasks vs DINO[^1].

**The DINO objective biases towards a <span class="case-term" tabindex="0" data-tooltip="Case 2: Scaling doesn't learn more/better lower level abstractions, which inhibits the quality and quantity of higher level semantics you can learn.">Case 2 failure</span>:** 

<div class="case-diagram">
  <img src="/images/posts/scaling-law-for-pure-vision/weak-bottom.png" alt="Weak bottom of the abstraction hierarchy">
</div>

The drivers for this failure case are more intricate than the MAE. DINO pools image information into a CLS token, which is then trained to match the CLS tokens between many views of the same image. 

- **What the learning signal is:** Invariance based methods like DINO work by shaping the learning signal to suppress information from the data. A lot of the suppressed information is lower level. 
  - Needing to predict a signal stripped of lower level abstractions allows the model to learn better top semantics than the MAE, however such a learning signal also underspecifies lower level abstractions. 
  - More subtly, stripping lower level abstractions from a prediction task may remove learning signal that's useful for for learning higher level semantics. Chalk lines of a math proof are a lower level abstraction, but needing to predict them is a learning signal for the concept of the math proof. If you suppress chalk lines from the prediction target, it's unlikely that you'll learn the math proof. 
- **How the model receives the learning signal:** In DINO, the model interfaces with the learning signal via the CLS token. The CLS token is the result of all semantic compositions the model performs, pooled into a single token. The gradients have to flow through the CLS token back to the patch representations, then back through the entire depth of the model, in order to assign credit to the semantic compositions. 

  This interface was not designed for utilising signals helpful for learning lower level abstractions, as the CLS interface makes credit assignment hard. 

Empirically, prolonged training of DINO starts hurting performance on dense tasks[^2]. The DINO objective is biased against lower level abstractions (at least in the final abstraction). 

DINO Caveats:
  Global tasks still improve with more training[^2]. This does not contradict with my claim of Case 2 failures. It'll be hard to learn the high level semantics that require preserved lower level abstractions. 
  
  Sticking with the math proof thought experiment, you need to preserve the lower level abstractions of some symbols late into the model. Some symbols may be meaningfully composed only after other symbols form semantic concepts. If you don't preserve these symbols until the semantic concepts are formed, you'll never learn the entire concept of said math proof. This is a statement about preserving access to lower level abstractions during composition, **different** from preserving lower level abstractions in the learning signal. 

## Acknowledgements

Thank you to (James) Wu Changyang for the graphics, Manu Gaur and Leo (maybe not his real name) for feedback on the first version of this post, and Akarsh Kumar for a conversation which motivated for more precision in some arguments. 

This post is also a love letter to the platonic representation hypothesis, which refers to the same Anna Karenina quote. 


## References

[^1]: J. Zhu, J. Qi, M. Ding, et al., "Understanding Self-Supervised Pretraining with Part-Aware Representation Learning," arXiv:2301.11915, 2023. [arXiv:2301.11915](https://arxiv.org/abs/2301.11915)
[^2]: O. Siméoni, H. V. Vo, M. Seitzer, et al., "DINOv3," arXiv:2508.10104, 2025. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)
