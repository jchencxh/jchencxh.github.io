---
title: "Constructive Self-Supervised Learning (Part 1): Designing generalisable deep self-supervision, and predicting lower-level abstractions for better semantics."
date: 2026-03-19 09:00:00 -0400
categories:
  - blog
---

## Contents
{: .no_toc }

* Auto generated table of contents
{:toc}

## Introduction

Traditionally, predicting lower-level abstractions is treated as harmful to learning higher-level semantics, and mainstream deep learning rarely supervises how intermediate abstractions are formed. This post argues for the opposite approach: we should explicitly shape the abstraction hierarchy during learning, and we should learn representations using signal from multiple levels of that hierarchy.

I call this family of objectives **constructive** SSL, because it explicitly supervises semantic construction. As a concrete example, I introduce cI-JEPA, a deeply supervised variant of I-JEPA[^1] in which a small set of student depths predicts a hierarchy of teacher representations rather than a single final target. By changing how this hierarchy is weighted, we can control the tradeoff between retaining lower-level structure and composing toward higher-level abstractions.

On ImageNet-100 at ViT-B scale, this improves linear-probe accuracy over the I-JEPA baseline. More broadly, I will argue two things: first, predicting an abstraction hierarchy is a useful way to design deep self-supervision for intermediate representations; second, using that same hierarchy to shape the final representation improves the final representation as well.

I perform my experiments on ImageNet-100 and ViT-B scale, and evaluate via improvement in linear probing accuracy on ImageNet-100.

The data and code can be found [here](https://drive.google.com/drive/u/0/folders/1pEw8WSDtMHG9NIVNmxEanrvXZd-9mStj). The code is designed to be run on a single 80GB A100/H100 on Google Colab. 

## Principles for learning lower level abstractions (with deep self-supervision).

### Setting a problem statement. 

Let’s assume we want to supervise semantic construction: shape intermediate abstractions as they form, not just the final
representation. In modern deep nets, that usually means designing deep objectives that act on hidden representations.

<div markdown="1" style="margin: 1.5rem 0; padding: 1rem 1.25rem; border: 1px solid var(--global-border-color); border-radius: 0;
text-indent: 0;">
**The problem statement for deep self-supervision:**

When learning a representation, we want hidden states to support three things: retaining useful lower-level abstractions, composing
them into higher-level ones, and dispersing what should no longer occupy capacity.

Too much dispersion throws away building blocks that may be useful later. Too much retention leaves too little capacity for new
abstractions. Too little composition leaves the final representation under-abstracted. A good deep objective should balance all
three.
</div>

A standard supervised deep loss, such as attaching a classifier to an intermediate layer, is poorly matched to this goal. It rewards whatever features solve the task immediately, including shortcuts, rather than abstractions that remain useful for later composition. That makes it a weak objective for shaping intermediate representations, especially on noisy natural data.

So for a given hidden representation, the deep objective should do two things: control the retention-dispersion tradeoff, and still
bias learning toward higher-level composition. If there was sufficient communication bandwidth between the levels of abstraction (there isn't in existing architectures), we wouldn't have to keep all our building blocks in a single set of latents, and the purpose of deep supervision becomes largely to bias towards non-spurious higher-level composition.


### Predicting an abstraction hierarchy is a good (deep) objective. 

When representations are learned through prediction, the target level determines what the model is pushed to keep and compose. Higher-level targets bias learning toward composing higher-level abstractions, while lower-level targets bias learning toward retaining lower-level detail. Predicting a hierarchy, rather than a single level, therefore gives us a way to control this tradeoff across hidden representations.

A hierarchy target also makes shortcut solutions less attractive. If the loss only has to match one very high-level, low-bit target, spurious solutions can satisfy it more easily. Requiring a representation to explain multiple levels of abstraction imposes more semantic constraints, so the model is pushed toward compositions that remain useful across the hierarchy rather than for a single target alone.

This is still imperfect. Predicting lower-level targets is only an indirect way to preserve lower-level structure, and standard residual architectures are not efficient for routing across different levels of abstraction. 

Predicting all the representations where all levels of abstraction are going to be learnt also provides conveniences for bootstrapping. For a given abstraction that we learn, the signal for learning it (i.e., the parts of the target hierarchy it’s supposed to be predicting) first forms dispersed throughout the network. We don’t know where exactly the ideal targets sit, and so we just predict everything. You will see concretely how this works in cI-JEPA. 


## Why predicting a hierarchy is generally a good objective

### What traditional SSL is doing

Higher-level abstractions are usually more useful for downstream tasks than raw low-level detail. If the learning objective asks a model to predict very low-level targets such as pixels (see, MAE[^3]), the model has to discover those higher-level abstractions indirectly. That is hard because the gap between the target and the abstractions we care about is large, and the resulting learning signal is noisy and sensitive to nuisance variation.

Latent SSL reduces this gap by predicting bootstrapped representations that are already biased toward higher-level structure rather than raw low-level targets. In effect, it lifts the level of abstraction at which supervision happens. Another way to describe this is that latent SSL works partly by dispersing lower-level detail. By not forcing the representation to preserve every low-level factor, it encourages the model to keep building blocks at a higher level of abstraction, which makes useful compositions easier to learn. The downside is that this mechanism is blunt with no controllability: it can also discard lower-level information that would still be useful for later composition or the final representation, and there are no knobs for us to use to tune how/what it discards. 


### Why mainstream SSL converged on dispersion, and why that is limiting


Mainstream latent SSL converged on dispersion largely because it lacks a direct way to encourage higher-level composition while still preserving useful lower-level detail. Dispersion is an effective shortcut: it suppresses low-level factors that are hard to compose and makes higher-level signals easier to learn. But it is also blunt. Some of the lower-level structure it throws away would still be useful for later composition or for the final representation. Constructive SSL is meant to replace that blunt tradeoff with explicit control over what gets retained, composed, and dispersed.


## The Actual cI-JEPA Algorithm

![cI-JEPA algorithm diagram](/images/posts/constructive-self-supervised-learning-part-1/ci-jepa-diagram.jpg)
*cI-JEPA algorithm visual, based off of Figure 3 from I-JEPA paper[^1].*

I train a standard ViT-B/16. 

The core change from I-JEPA to cI-JEPA is actually very small: instead of using a single source depth to predict a single teacher representation, cI-JEPA samples a small set of source depths extracted from all the depths collected from the student vision encoder (i.e., we take early exits after every block), and asks each depth from the small source set to predict all the collected depths from the EMA teacher. 

Everything not mentioned here follows I-JEPA. In particular, I keep the same context/target-block masking scheme, the same target-location-conditioned predictor setup, and the same EMA teacher--student asymmetry. This section only describes the representation-level cI-JEPA objective

In the scaled-down setting used here, I train on ImageNet-100 (about a tenth of ImageNet-1000 size) for 200 epochs by default, and always use a LR warmup covering roughly <span>&#92;(2.5&#92;%&#92;)</span> of total training steps. 

Note that my setup also removes the final encoder LayerNorm (this has little effect on the baseline accuracy and is mostly done because there are no deep LayerNorms), uses constant weight decay <span>&#92;(0.05 &#92;)</span>, as well as use RoPE instead of sincos positional embeddings (as more recent JEPAs use RoPE). The goal is to keep the optimization and masking recipe as close as possible to I-JEPA while changing only the supervision objective to minimize confounds. 

### Collected depths and teacher targets

Let the collected encoder depths be

$$
\mathcal D = \{d_1,\dots,d_{11}\}.
$$


In the experiments discussed here we use a ViT-B which has 12 transformer blocks, and unless otherwise noted we sample from the representations that come out of the deepest 11 blocks, so these are blocks <span>&#92;(1,&#92;dots,11&#92;)</span>. Note that <span>&#92;(d_{11}&#92;)</span> corresponds to the final output of the ViT-B. 

The choice to not include the representation after the very first block in <span>&#92;(&#92;mathcal D&#92;)</span> was accidental. I did not see a reason why this would affect the results or discussion so I didn't re-run the experiments, but future work should probably include it (i.e., have  <span>&#92;(&#92;mathcal D = &#92;{d_0, &#92;dots, d_{11}&#92;} &#92;)</span> ).

For an image <span>&#92;(x&#92;)</span>, I sample context patches <span>&#92;(C&#92;)</span> and masked target blocks <span>&#92;(&#92;{T_m&#92;}_{m=1}^M&#92;)</span> exactly as in I-JEPA. The student encoder <span>&#92;(f_&#92;theta&#92;)</span> only processes the context patches, while the EMA teacher <span>&#92;(f_&#92;xi&#92;)</span> always processes the full image and provides stop-gradient targets:

$$
z_s^C = f_\theta^{(s)}(x_C),
\qquad
y_t = \operatorname{sg}\!\bigl(f_\xi^{(t)}(x)\bigr),
\qquad
\xi \leftarrow m\xi + (1-m)\theta,
$$

where <span>&#92;(s,t &#92;in &#92;mathcal D&#92;)</span>, <span>&#92;(f^{(s)}&#92;)</span> denotes the hidden state at collected depth <span>&#92;(s&#92;)</span>, and <span>&#92;(&#92;operatorname{sg}(&#92;cdot)&#92;)</span> denotes stop-gradient.

Note that under standard I-JEPA with a ViT-B, the collected encoder depths are

$$
\mathcal D = \{d_{11}\}.
$$

### Sampling source depths

A useful way to view cI-JEPA is as a **source-depth <span>&#92;(&#92;times&#92;)</span> target-depth** table of prediction problems. Rows correspond to student source depths, and columns correspond to teacher target depths.

Unlike vanilla I-JEPA, cI-JEPA does not use all depths collected in &#92;( &#92;mathcal D &#92;) as supervised rows at every step. Instead, I sample some number (two by default) of random intermediate rows and always include the deepest row:

$$
S = \{d_a, d_b, d_{11}\},
\qquad
d_a,d_b \sim \operatorname{Unif}(\mathcal D \setminus \{d_{11}\}),
\qquad
d_a \neq d_b.
$$

So each optimization step supervises exactly three source depths: two random intermediate depths plus the final depth. Over training, all intermediate depths are revisited, but each step only pays for three source rows. 

This sampling was mostly a training efficiency consideration. I did not ablate different choices for how many intermediate rows I sample each time during deep supervision due to compute constraints (this is a personal project), and always just sampled two when deep supervision is being done. 

### Depth-specific predictor pathways

Each sampled source depth <span>&#92;(s &#92;in S&#92;)</span> has its own predictor pathway <span>&#92;(P_s&#92;)</span>. Each predictor &#92;(P_s&#92;) is a separate instance of the narrow ViT that the I-JEPA uses. 


It uses only the student context representation at that depth, together with the target block coordinates, to predict the masked block. In the implementation used here, these predictor pathways are fully separate rather than shared.

The predictor output is then mapped into each teacher depth with a bank of source--target-specific linear heads <span>&#92;(&#92;{H_{s&#92;to t}&#92;}_{t&#92;in&#92;mathcal D}&#92;)</span>:

$$
h_{s,m} = P_s(z_s^C, C, T_m),
\qquad
\hat y_{s\to t}^{\,T_m} = H_{s\to t}(h_{s,m}).
$$

The important point is that there is **no fusion across source depths**: each sampled depth must, by itself, predict the entire collected teacher hierarchy on the masked target block.

### Multi-depth masked latent objective

The cI-JEPA objective is

$$
\mathcal L_{\text{cI-JEPA}}
=
\frac{1}{M|S|}
\sum_{m=1}^{M}
\sum_{s\in S}
\sum_{t\in\mathcal D}
w_{s,t}\;
\ell\!\left(\hat y_{s\to t}^{\,T_m},\, y_t^{\,T_m}\right),
$$

where <span>&#92;(&#92;ell&#92;)</span> is mean-squared error over the masked target tokens, and <span>&#92;(y_t^{&#92;,T_m}&#92;)</span> denotes the teacher features at depth <span>&#92;(t&#92;)</span> for the target block <span>&#92;(T_m&#92;)</span>.

In this setup, every sampled source depth predicts **all** target depths. The remaining design choice is therefore how to weight the target columns within each source row.

### Biasing supervision toward the deepest target

Instead, I bias every row toward the deepest teacher target with `all_to_last_weight`, and I can further bias the final source row with `last_to_last_weight`.

Let

$$
\alpha = \texttt{all_to_last_weight},
\qquad
\beta = \texttt{last_to_last_weight}.
$$

Then the row-wise target weights are

$$
w_{s,t}=
\begin{cases}
\beta, & s=d_{11},\; t=d_{11}, \\[6pt]
\dfrac{1-\beta}{L-1}, & s=d_{11},\; t\neq d_{11}, \\[12pt]
\alpha, & s\neq d_{11},\; t=d_{11}, \\[6pt]
\dfrac{1-\alpha}{L-1}, & s\neq d_{11},\; t\neq d_{11}.
\end{cases}
$$

Because each row sums to one, these weights change **where** a row places its loss mass without changing the overall contribution of that row.

For example, when we have <span>&#92;(&#92;alpha=0.5&#92;)</span> and <span>&#92;(&#92;beta=0.8&#92;)</span> with <span>&#92;(L=11&#92;)</span>:

- every **non-final** source row places <span>&#92;(50&#92;%&#92;)</span> of its weight on the deepest teacher target and <span>&#92;(5&#92;%&#92;)</span> on each of the other <span>&#92;(10&#92;)</span> depths;
- the **final** source row places <span>&#92;(80&#92;%&#92;)</span> of its weight on the deepest teacher target and <span>&#92;(2&#92;%&#92;)</span> on each of the other <span>&#92;(10&#92;)</span> depths.

Intuitively, this makes the deepest EMA representation the anchor of the objective. This deepest representation corresponds to the highest level of abstraction, and so by increasing <span>&#92;(&#92;alpha, &#92;beta&#92;)</span>, you can bias the representations at a given source row towards higher level composition while still being grounded by the entire abstraction heirachy. 

As an implementation note, in the code, `all_to_last_weight` biases every source depth (including the last source depth) toward the last target depth, while `last_to_last_weight` only biases the last source depth toward the last target and leaves non-last source depths uniform. In the code, if `last_to_last_weight`is set, it will overwrite `all_to_last_weight` in the code.


### cI-JEPA summary

One training step can be written as:

1. **Sample masks as in I-JEPA.**

   $$ 
   C,\; \{T_m\}_{m=1}^M \sim \text{MaskSampler}(x).
   $$

2. **Sample the supervised source depths.**

   $$ 
   S = \{d_a,d_b,d_{11}\},
   \qquad
   d_a,d_b \sim \operatorname{Unif}(\mathcal D \setminus \{d_{11}\}),
   \qquad
   d_a \neq d_b.
   $$

3. **Run the student on the context view only.**

   $$ 
   \{z_s^C\}_{s\in\mathcal D}
   =
   f_\theta(x_C).
   $$

4. **Run the EMA teacher on the full image.**

   $$ 
   \{y_t\}_{t\in\mathcal D}
   =
   \operatorname{sg}\!\bigl(f_\xi(x)\bigr).
   $$

5. **For each target block and each sampled source depth, predict the full teacher hierarchy.**

   For every <span>&#92;(m &#92;in &#92;{1,&#92;dots,M&#92;}&#92;)</span>, every <span>&#92;(s &#92;in S&#92;)</span>, and every <span>&#92;(t &#92;in &#92;mathcal D&#92;)</span>,

   $$ 
   h_{s,m} = P_s(z_s^C, C, T_m),
   \qquad
   \hat y_{s\to t}^{\,T_m} = H_{s\to t}(h_{s,m}).
   $$

6. **Accumulate the weighted multi-depth masked latent loss.**

   $$ 
   \mathcal L_{\text{cI-JEPA}}
   =
   \frac{1}{M|S|}
   \sum_{m=1}^{M}
   \sum_{s\in S}
   \sum_{t\in\mathcal D}
   w_{s,t}\;
   \ell\!\left(\hat y_{s\to t}^{\,T_m},\, y_t^{\,T_m}\right).
   $$

7. **Update the student, then update the EMA teacher.**

   $$ 
   \theta \leftarrow \theta - \eta \nabla_\theta \mathcal L_{\text{cI-JEPA}},
   \qquad
   \xi \leftarrow m\xi + (1-m)\theta.
   $$

### Evaluation methodology

I evaluate learned representations with a linear-probing protocol closely matching the one used in I-JEPA, but adapted to ImageNet-100 and optimized with AdamW rather than LARS. After pretraining, I freeze the encoder and train a single linear classifier on top of the final-layer representation. Since the encoder has no CLS token, I extract the final patch tokens, average-pool them across spatial locations, and feed the pooled feature into a linear layer producing 100-way logits. When using EMA pretraining, I probe the EMA teacher encoder rather than the online student. I report top-1 accuracy on the ImageNet-100 validation set.

The data transforms follow the standard I-JEPA-style linear-evaluation recipe. However, unlike the larger-budget linear probe used in the original I-JEPA work, I use a much smaller probe budget. The probe classifier is trained for only 10 epochs with batch size 256 using AdamW. 

### Exact probe hyperparameters

- **Dataset:** ImageNet-100
- **Encoder used for probing:** frozen EMA teacher encoder
- **Representation:** final-layer patch tokens, mean-pooled over spatial positions
- **Probe head:** single linear layer from embedding dimension to 100 classes
- **Optimizer:** AdamW
- **Learning rate:** `3e-3`
- **Weight decay:** `0.0`
- **Epochs and LR schedule:** linear decay from `3e-3` to `0.0` over 10 epochs
- **Train batch size:** `256`
- **Validation batch size:** `256`
- **Probe train transform:** `RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=BICUBIC)` + `RandomHorizontalFlip(0.5)` + ImageNet normalization
- **Probe validation transform:** `Resize(256, interpolation=BICUBIC)` + `CenterCrop(224)` + ImageNet normalization
- **Number of classes:** `100`


### Results and discussion


| Run ID | Supervision Method | Epochs | <span>&#92;(&#92;alpha&#92;)</span> | <span>&#92;(&#92;beta&#92;)</span> | Top-1 accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `L200` | I-JEPA baseline <span>&#92;((&#92;mathcal D = &#92;{d_{11}&#92;})&#92;)</span> | 200  | N/A | 1.0 | 64.04 % |
| `L300` | I-JEPA baseline <span>&#92;((&#92;mathcal D = &#92;{d_{11}&#92;})&#92;)</span> | 300 | N/A | 1.0 | 66.96 % |
| `R-U` | cI-JEPA, uniform weighting/no biasing <span>&#92;((&#92;mathcal D = &#92;{d_1, &#92;dots, d_{11}&#92;})&#92;)</span> | 200 | <span>&#92;(&#92;frac{1}{11}&#92;)</span> (uniform) | <span>&#92;(&#92;frac{1}{11}&#92;)</span> (uniform) | 67.34 % |
| `R-A05-B05` | cI-JEPA, high all bias <span>&#92;((&#92;mathcal D = &#92;{d_1, &#92;dots, d_{11}&#92;})&#92;)</span> | 200 | 0.5 | 0.5 | 68.96 % |
| `R-A08-B08` | cI-JEPA, higher all bias <span>&#92;((&#92;mathcal D = &#92;{d_1, &#92;dots, d_{11}&#92;})&#92;)</span> | 200 | 0.8 | 0.8 | 66.94 % |
| `R-A05-B08` | cI-JEPA, high intermediate + higher final bias <span>&#92;((&#92;mathcal D = &#92;{d_1, &#92;dots, d_{11}&#92;})&#92;)</span>| 200 | 0.5 | 0.8 | **70.06 %** |
| `R-A05-B08-12` | cI-JEPA, high intermediate + higher final bias (depth 0 added) <span>&#92;((&#92;mathcal D = &#92;{d_0, &#92;dots, d_{11}&#92;})&#92;)</span>| 200 | 0.5 | 0.8 | 69.1% |
| `R-B05` | cI-JEPA, high final bias <span>&#92;((&#92;mathcal D = &#92;{d_1, &#92;dots, d_{11}&#92;})&#92;)</span> | 200 | <span>&#92;(&#92;frac{1}{11}&#92;)</span> (uniform) | 0.5 | 68.42 % |
| `R-B08` | cI-JEPA, higher final bias <span>&#92;((&#92;mathcal D = &#92;{d_1, &#92;dots, d_{11}&#92;})&#92;)</span> | 200 | <span>&#92;(&#92;frac{1}{11}&#92;)</span> (uniform) | 0.8 | 69.36 % |
| `R-A05-B10` | cI-JEPA, high intermediate bias + final doesn't predict hierarchy <span>&#92;((&#92;mathcal D = &#92;{d_1, &#92;dots, d_{11}&#92;})&#92;)</span> | 200 | 0.5 | 1.0 | 69.22 % |
| `R-A.-B05` | cI-JEPA, no deep supervision + high final bias <span>&#92;((&#92;mathcal D = &#92;{d_1, &#92;dots, d_{11}&#92;})&#92;)</span> | 200 | N/A | 0.5 | 64.66 % |
| `R-A.-B08` | cI-JEPA, no deep supervision + higher final bias <span>&#92;((&#92;mathcal D = &#92;{d_1, &#92;dots, d_{11}&#92;})&#92;)</span> | 200 | N/A | 0.8 | 66.02 % |


As a reminder:
1. A higher <span>&#92;(&#92;alpha&#92;)</span> means that non-last depth sources will weigh their prediction of the last depth target higher. 
2. A higher <span>&#92;(&#92;beta&#92;)</span> means that ONLY the last depth will weigh its prediction of the last depth target higher. 
3. Further, remember that in the actual code, `all_to_last_weight` and `last_to_last_weight` don't exactly correspond to <span>&#92;(&#92;alpha&#92;)</span> and <span>&#92;(&#92;beta&#92;)</span>. In the code, if `last_to_last_weight` isn't set, <span>&#92;(&#92;beta&#92;)</span> will default to <span>&#92;(&#92;alpha&#92;)</span>. 

**The important observations:**
1. Run `R-A05-B10` which performs deep supervision, while the final representation is shaped by predicting **only** the target's final representation, still outperforms the vanilla baselines `L200` and `L300`. This shows that shaping intermediate representations by predicting a target abstraction hierarchy learns good intermediate abstractions, which leads to a better final representation.

2. Run `R-A05-B08` which shapes the final representation by using it to predict lower level abstractions beats `R-A05-B10` which does deep supervision but does not use the final representation to predict lower level representations. This shows that shaping the final representation by predicting lower level abstractions can boost its semantic performance.

3. Comparing runs `R-U`, `R-A05-B05`, `R-A08-B08`, and `R-A05-B08`: There is a balance to be made by tuning <span>&#92;(&#92;alpha&#92;)</span> and <span>&#92;(&#92;beta&#92;)</span> to bias the representations to compose to higher level abstractions and retaining lower level abstractions that are more predictive of lower levels of the hierarchy. Dispersing too much (`R-A08-B08`) starts hurting. Not composing higher/dispersing too little (`R-U`) doesn't learn enough semantics. 

    We don't want to disperse too much in the middle of the network so we can retain more factors potentially useful for further semantic composition, while we want to disperse more and compose more in the final block so that we can compose the more higher level abstractions (`R-A05-B08` is better than `R-A05-B05`). 

4. `R-B05` and `R-B08`: even without biasing the deep supervision towards a higher level, biasing the final representation towards higher levels improves semantics. 

5. `R-A.-B05` and `R-A.-B08`: without deep supervision predicting the hierarchy with the final representation produces noisy targets. Conversely, deep supervision produces a good abstraction hierarchy that is effective for defining a prediction objective on. 

6. As a sanity check, I also added the first block's output (<span>&#92;(d_0&#92;)</span>) for run `R-A05-B08-12`. There's a slight performance dip likely because predicting the first representation does not provide much learning signal. 

Is this algorithm efficient? No. But I certainly hope it's illustrative. 

<!-- ## Why / How I Think It Works (YOU SHOULD READ THIS) -->


## Current SSL is probably doing some learning over the entire hierarchy of abstractions

I think it’s also important to consider the possibility that current latent SSL methods are already somewhat learning over intermediate abstractions in the target, even if not explicitly doing deep supervision. 

When people talk about “more pixel space information” vs “more semantic information” inside a representation, it’s useful to think of this as a weighted window on the spectrum of pixels to semantics, with higher absolute weighting on the spectrum meaning that the signals associated with that abstraction level are less dispersed/more easily recoverable from the representation (e.g., via a linear probe).

Through this writing, for rhetorical purposes, I often discretize this weighted window idea and refer to a single instance of an uneven weighted window as “a (single) level of abstraction”, even though it’s not entirely accurate. At any given "high-level" representation, we often retain some lower level abstractions.

Pixels are not (to a good extent) linearly recoverable from SSL latents trained with I-JEPA or DINO[^2], but the RAE paper[^8] shows that pixels are recoverable from SSL latents with enough effort (not linearly). This would be consistent with pixel-space information being dispersed in SSL latents. It likely also follows that intermediate abstractions are dispersed in SSL latents too, with the higher levels just being less dispersed. Using our weighted window analogy, current SSL representations have lower level abstractions less weighted, but still represent the entire hierarchy.

In DINO, the training target is a teacher network’s softmax distribution (derived from its final representations via a projection head), so the supervision signal is ultimately bootstrapped from representations learned by the model itself. As said representations likely contain information about lower/intermediate abstractions, the target distribution likely does too (i.e., the target contains information about the entire abstraction hierarchy). Further, note that most modern neural nets have a residual connection, which is quite a natural bias towards retaining abstractions.

So, an interpretation of a factor behind the existing success of latent SSL may be that they already learn over the entire abstraction hierarchy.

It’s just that these lower level abstractions are worse and more dispersed, as their construction isn’t supervised, and we don’t provide some explicit target for them (and we can’t because they’re too dispersed).

Under constructive SSL, we learn better lower level compositions by supervising construction, and thus learn better lower level abstractions. This allows us to more explicitly specify a prediction objective over the entire abstraction hierarchy.


## Some closing thoughts (on JEPAs)

Constructive SSL is a position about doing SSL. With current deep learning architectures, it will often use deep supervision; but I do not want to limit your imagination to doing deep supervision on existing architectures. The main intent of this blog post is an attempt at distilling the core ideas of what I think works for learning abstractions, not some more specific algorithm or implementation.

The choice of converting I-JEPA into cI-JEPA comes from the intuition that predicting representations directly provides the most “raw” signal for a given level of abstraction, and how they differ from other levels of abstraction. Further, working off of a well-known existing design makes the idea significantly more communicable, as well as provides a good baseline, to/for people.

Other choices like choosing instead to predict some distribution over prototypes like DINO may work, but they obscure the signal of the raw representation latents: DINO supervises a normalized prototype-assignment distribution (softmax outputs), which obscures information from the raw representation latents.

Though I’m not entirely sure of the current direction of regularization-driven methods for JEPAs like LeJEPA, probably in part because I don’t understand them perfectly, the core intuition of using JEPAs to lift the level of abstraction we supervise at is very powerful. Here, we just lift to many differing levels of abstraction.

I also wouldn’t limit your imagination to doing just JEPAs. The important intuition that I’m trying to convey to you is simply that you should supervise the semantic construction of abstractions explicitly and learn by predicting varying levels of abstraction, and not just rely on an internal barely-constrained representation search while explicitly learning over a single level of abstraction (this applies to both methods like the MAE[^3] or I-JEPA).

Another intention of this design is to scale vision models. Scalable methods for intelligence have some notion of “more computing power can build/discover more abstractions from data”. Traditional latent SSL having a dispersion bias almost feels anti-scalable to me, as it throws away building blocks and potentially useful learning signals.

V-JEPA 2.1[^4] has a deep supervision setup that’s closest to cI-JEPA from what I am aware of. They fix the layers they’re supervising, as well as fuse the different intermediate representations by the channel for prediction, and predict many levels with a fused representation. I do think that the authors under-emphasised the interestingness of doing deep supervision. At the risk of having a bit too much hubris, I think ideas I’ve presented are the factors that underlie why the V-JEPA 2.1 deep supervision works too.

I thought about calling my cI-JEPA design I-JEPA 1.1 because I think it’s a much more fitting name, but I did not want to impose a version number onto the original authors.

I would also not limit your imagination to vision or JEPAs. The idea behind doing representation learning more constructively is broadly applicable to everything. It is much more a way of thinking about how to learn good abstractions of reality than it is a specific vision (or even JEPA) thing.

I did not refer to constructive methods as JEPAs because it implies that you have to be predicting something, which may not always be the case. At least, you just want signals from the entire abstraction hierarchy that arises from nature, and explicitly shape the entire abstraction hierarchy. Though, it could turn out that all good constructive SSL objectives are JEPAs (i.e., you predict) anyways.

An original desire for this project was to find a way to represent an abstraction with the most efficient circuitry (least expressive power) possible. This and other ideas will hopefully be in part 2, where I talk a bit more about the interesting things you can do with a constructive objective.

## Acknowledgements

Dominik and Minqi, and their LAPO paper[^5], was one of the initial inspirations for this project. When I was just starting out, Dominik said something to me like “just capture everything”. It was pretty nebulous to me what this meant at the time, but my current interpretation is something along the lines of “capture all the abstractions”.

Akarsh and Kenneth, their work on UFR/FER representations[^6] and evolutionary methods, as well as discussions we’ve had, helped shape and encourage some of these ideas.

Saining and Philip helped prompt some of these ideas.

Another initial inspiration of this project started out involving EBMs along with some guidance from Yilun[^7]. I had (and have) a minor obsession with composing abstractions (with EBMs) which I couldn’t get working with existing methods, which led to this.

James, Samson, and Cem helped me pick out the naming scheme.

## References

[^1]: M. Assran, Q. Duval, I. Misra, et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture," arXiv:2301.08243, 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
[^2]: M. Caron, H. Touvron, I. Misra, et al., "Emerging Properties in Self-Supervised Vision Transformers," arXiv:2104.14294, 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
[^3]: K. He, X. Chen, S. Xie, et al., "Masked Autoencoders Are Scalable Vision Learners," arXiv:2111.06377, 2021. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
[^4]: M. Assran, A. Bardes, D. Fan, et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning," arXiv:2506.09985, 2025. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
[^5]: D. Schmidt, M. Jiang, "Learning to Act without Actions," arXiv:2312.10812, 2023. [arXiv:2312.10812](https://arxiv.org/abs/2312.10812)
[^6]: A. Kumar, J. Clune, J. Lehman, K. O. Stanley, "Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis," arXiv:2505.11581, 2025. [arXiv:2505.11581](https://arxiv.org/abs/2505.11581)
[^7]: Y. Du, S. Li, I. Mordatch, "Compositional Visual Generation and Inference with Energy Based Models," arXiv:2004.06030, 2020. [arXiv:2004.06030](https://arxiv.org/abs/2004.06030)
[^8]: B. Zheng, N. Ma, S. Tong, S. Xie, "Diffusion Transformers with Representation Autoencoders," arXiv:2510.11690, 2025. [arXiv:2510.11690](https://arxiv.org/abs/2510.11690)
