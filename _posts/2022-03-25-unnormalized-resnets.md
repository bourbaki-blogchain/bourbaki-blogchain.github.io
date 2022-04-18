---
layout: post
title: Normalization is dead, long live normalization!
tags: [normalization, initialization, propagation, skip connections, residual networks]
authors: Hoedt, Pieter-Jan; Hochreiter, Sepp; Klambauer, Günter
---

Since the advent of Batch Normalization (BN), almost every state-of-the-art (SOTA) method uses some form of normalization.
After all, normalization generally speeds up learning and leads to models that generalize better than their unnormalized counterparts.
This turns out to be especially useful when using some form of skip connections, which are prominent in Residual Networks (ResNets), for example.
However, [Brock et al. (2021a)](#brock21characterizing) suggest that SOTA performance can also be achieved using **ResNets without normalization**!

The fact that Brock et al. went out of their way to get rid of something as simple as BN in ResNets, for which BN happens to be especially helpful, does raise a few questions:

 1. Why get rid of BN in the first place[?](#alternatives)
 2. How (easy is it) to get rid of BN in ResNets[?](#moment-control)
 3. Is BN going to become obsolete in the near future[?](#limitations)
 4. Does this allow us to gain insights into why BN works so well[?](#insights)
 5. Wait a second... Are they getting rid of normalization or just BN[?](#conclusion)

The goal of this blog post is to provide some insights w.r.t. these questions using the results from [Brock et al. (2021a)](#brock21characterizing).

<style>
    figcaption { color: gray; }
</style>

## Contents

 - [Normalization](#normalization)
    * [Origins](#origins)
    * [Batch Normalization](#batch-normalization)
    * [Alternatives](#alternatives)
 - [Skip Connections](#skip-connections)
    * [History](#history)
    * [Moment Control](#moment-control)
 - [Normalizer-Free ResNets](#normalizer-free-resnets)
    * [Old Ideas](#old-ideas)
    * [Imitating Signal Propagation](#imitating-signal-propagation)
    * [Performance](#performance)
 - [Discussion](#discussion)
    * [Limitations](#limitations)
    * [Insights](#insights)
    * [Conclusion](#conclusion)
 - [Extra Code Snippets](#extra-code-snippets)
 - [References](#references)


## Normalization

To set the scene for a world without normalization, we start with an overview of normalization layers in neural networks.
Batch Normalization is probably the most well-known method, but there are plenty of alternatives.
Despite the variety of normalization methods, they all build on the same principle ideas.

### Origins

The design of modern normalization layers in neural networks is mainly inspired by data normalization ([Lecun et al., 1998](#lecun98efficient); [Schraudolph, 1998](#schraudolph98centering); [Ioffe & Szegedy, 2015](#ioffe15batchnorm)).
In the setting of a simple linear regression, it can be shown (see e.g., [Lecun et al., 1998](#lecun98efficient)) that the second-order derivative, i.e., the Hessian, of the objective is exactly the covariance of the input data, $\mathcal{D}$:

$$\frac{1}{|\mathcal{D}|} \sum_{(\boldsymbol{x}, y) \in \mathcal{D}} \nabla_{\boldsymbol{w}}^2 \frac{1}{2}(\boldsymbol{w}^\mathsf{T} \boldsymbol{x} - y)^2 = \frac{1}{|\mathcal{D}|}  \sum_{(\boldsymbol{x}, y) \in \mathcal{D}}\boldsymbol{x} \boldsymbol{x}^\mathsf{T}.$$

If the Hessian of an optimization problem is (close to) the identity, it becomes much easier to find a solution ([Lecun et al., 1998](#lecun98efficient)).
Therefore, learning should become easier if the input data is whitened &mdash; i.e., is transformed to have an identity covariance matrix.
However, full whitening of the data is often costly and might even degenerate generalization performance ([Wadia et al., 2021](#wadia21whitening)).
Instead, the data is _normalized_ to have zero mean and unit variance to get at least some of the benefits of an identity Hessian.

When considering multi-layer networks, the expectation would be that things get more complicated.
However, it turns out that the benefits of normalizing the input data for linear regression directly carry over to the individual layers of a multi-layer network ([Lecun et al., 1998](#lecun98efficient)).
Therefore, simply normalizing the inputs to a layer &mdash; i.e., the outputs from the previous layer &mdash; should also help to speed up the optimization of the weights in that layer.
Using these insights, [Schraudolph (1998)](#schraudolph98centering) showed empirically that centering the activations effectively speeds up learning.

Also initialization strategies commonly build on these principles (e.g., [Lecun et al., 1998](#lecun98efficient); [Glorot & Bengio, 2010](#glorot10understanding); [He et al., 2015](#he15delving)).
Since the initial parameters of a layer are independent of the inputs, they can easily be tuned.
When tuned correctly, it can be assured that the (pre)-activations of each layer are normalized throughout the network before the first update.
However, as soon as the network is being updated, the distributions change and the normalizing properties of the initialization get lost ([Ioffe & Szegedy, 2015](#ioffe15batchnorm)).

### Batch Normalization

In contrast to classical initialization methods, Batch Normalization (BN) is able to maintain fixed mean and variance of the activations as the network is being updated ([Ioffe & Szegedy, 2015](#ioffe15batchnorm)).
Concretely, this is achieved by applying a typical data normalization to every mini-batch of data, $\mathcal{B}$:

$$\hat{\boldsymbol{x}} = \frac{\boldsymbol{x} - \boldsymbol{\mu}_\mathcal{B}}{\boldsymbol{\sigma}_\mathcal{B}}.$$

Here $\boldsymbol{\mu}\_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum\_{\boldsymbol{x} \in \mathcal{B}} \boldsymbol{x}$ is the mean over the inputs in the mini-batch and $\boldsymbol{\sigma}_\mathcal{B}$ is the corresponding standard deviation.
Also, note that the division is element-wise and generally is numerically stabilized by some $\varepsilon$ when implemented.
In case a zero mean and unit variance is not desired, it is also possible to apply an affine transformation $\boldsymbol{y} = \boldsymbol{\gamma} \odot \hat{\boldsymbol{x}} + \boldsymbol{\beta}$ with learnable scale $(\boldsymbol{\gamma})$ and mean ($\boldsymbol{\beta}$) parameters ([Ioffe & Szegedy, 2015](#ioffe15batchnorm)).
Putting these formulas together in ([PyTorch](https://pytorch.org)) code, BN can be summarized as follows:

```python
def batch_normalize(x, gamma=1., beta=0., eps=1e-5):
    mu = torch.mean(x, dim=(0, -1, -2))
    var = torch.var(x, dim=(0, -1, -2))
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return gamma * x_hat + beta
```

The above description explains the core operation of BN during training.
However, during inference, it is not uncommon to desire predictions for single samples.
Obviously, this would cause trouble because a mini-batch with a single sample has zero variance.
Therefore, it is common to accumulate the statistics that are used for normalization ( $\boldsymbol{\mu}\_\mathcal{B}$ and $\boldsymbol{\sigma}\_\mathcal{B}^2$ ) over multiple mini-batches during training.
These accumulated statistics can then be used as estimators for the mean and variance during inference.
This makes it possible for BN to be used on single samples during inference.

The original reason for introducing BN was to alleviate the so-called _internal covariate shift_, i.e. the change of distributions as the network updates.
More recent research has pointed out, however, that internal covariate shift does not necessarily deteriorate learning dynamics ([Santurkar et al., 2018](#santurkar18how)).
Apparently, [Ioffe & Szegedy (2015)](#ioffe15batchnorm) also realized that simply normalizing the signal does not suffice to achieve good performance: 

 > [...] the model blows up when the normalization parameters are computed outside the gradient descent step.

 All of this seems to indicate that part of the success of BN is due to the effects it has on the gradient signal.
 The affine transformation in BN simply scales the gradient, such that $\nabla_{\hat{\boldsymbol{x}}} \mathcal{L} = \boldsymbol{\gamma} \odot \nabla_{\boldsymbol{y}} \mathcal{L}.$
 The normalization operation, on the other hand, transforms the gradient, $\boldsymbol{g} = \nabla_{\hat{\boldsymbol{x}}} \mathcal{L}$, as follows:

 $$\nabla_{\boldsymbol{x}} \mathcal{L} = \frac{1}{\boldsymbol{\sigma}_\mathcal{B}} \big(\boldsymbol{g} - \mu_g \,\boldsymbol{1} - \operatorname{cov}(\boldsymbol{g}, \hat{\boldsymbol{x}}) \odot \hat{\boldsymbol{x}} \big),$$

where $\mu_g = \sum_{\boldsymbol{x} \in \mathcal{B}} \nabla_{\hat{\boldsymbol{x}}} \mathcal{L}$ and $\operatorname{cov}(\boldsymbol{g}, \hat{\boldsymbol{x}}) = \frac{1}{|\mathcal{B} |} \sum_{\boldsymbol{x} \in \mathcal{B}} \boldsymbol{g} \odot \hat{\boldsymbol{x}}.$
Note that this directly corresponds to centering the gradients, which is also supposed to improve learning speed ([Schraudolph, 1998](#schraudolph98centering)).

In the end, everyone seems to agree that one of the main benefits of BN is that it enables higher learning rates ([Ioffe & Szegedy, 2015](#ioffe15batchnorm); [Bjorck et al., 2018](#bjorck18understanding); [Santurkar et al., 2018](#santurkar18how); [Luo et al., 2019](#luo19towards)), which results in faster learning and better generalization.
An additional benefit is that BN is scale-invariant and therefore much less sensitive to weight initialization ([Ioffe & Szegedy, 2015](#ioffe15batchnorm); [Ioffe, 2017](#ioffe17batchrenorm)).

### Alternatives

Why would we ever want to get rid of BN then?
Although BN provides important benefits, it also comes with a few downsides:

 - BN does not work well with **small batch sizes** ([Ba et al., 2016](#ba16layernorm); [Salimans & Kingma, 2016](#salimans16weightnorm); [Ioffe, 2017](#ioffe17batchrenorm)).
   For a batch size of one, we have zero standard deviation, but also with a few samples, the estimated statistics are often not accurate enough.
 - BN is not directly applicable to certain input types ([Ba et al. 2016](#ba16layernorm); also see Figure&nbsp;[1](#fig_dims)) and performs poorly when there are **dependencies between samples** in a mini-batch ([Ioffe, 2017](#ioffe17batchrenorm)).
 - BN uses **different statistics for inference** than those used during training ([Ba et al., 2016](#ba16layernorm); [Ioffe, 2017](#ioffe17batchrenorm)).
   This is especially problematic if the distribution during inference is different or drifts away from the training distribution.
 - BN does not play well with **other regularization** methods ([Hoffer et al., 2018](#hoffer18norm)).
   This is especially known for $\mathrm{L}_2$-regularization ([Hoffer et al., 2018](#hoffer18norm)) and dropout ([Li et al., 2019](#li19understanding)).
 - BN introduces a significant **computational overhead** during training ([Ba et al., 2016](#ba16layernorm); [Salimans & Kingma, 2016](#salimans16weightnorm); [Gitman and Ginsburg, 2017](#gitman17comparison)).
   Because of the running averages, also memory requirements increase when introducing BN.

Therefore, alternative normalization methods have been proposed to solve one or more of the problems listed above while trying to maintain the benefits of BN.

<figure id="fig_dims">
    <img src="{{ site.url }}/public/images/2022-03-25-unnormalized-resnets/data_dimensions.svg" alt="visualization of different input data types">
    <figcaption>
        Figure&nbsp;1: Different input types in terms of their typical 
        batch size ($|\mathcal{B}|$), the number of channels/features ($C$) and the <em>size</em> of the signal ($S$) (e.g. width times height for images).
        Image inspired by (<a href="#wu18groupnorm">Wu & He, 2018</a>).
    </figcaption>
</figure>

One family of alternatives simply computes the statistics along different dimensions (see Figure&nbsp;[2](#fig_norm)).
**Layer Normalization (LN)** is probably the most prominent example in this category ([Ba et al., 2016](#ba16layernorm)).
Instead of computing the statistics over samples in a mini-batch, LN uses the statistics of the feature vector itself.
This makes LN invariant to weight shifts and scaling individual samples.
BN, on the other hand, is invariant to data shifts and scaling individual neurons.
LN generally outperforms BN in fully connected and recurrent networks but does not work well for convolutional architectures according to [Ba et al. (2016)](#ba16layernorm).
**Group Normalization (GN)** is a slightly modified version of LN that also works well for convolutional networks ([Wu et al., 2018](#wu18groupnorm)).
The idea of GN is to compute statistics over groups of features in the feature vector instead of all features.
For convolutional networks that should be invariant to changes in contrast, statistics can also be computed over single image channels for each sample.
This gives rise to a technique known as **Instance Normalization (IN)**, which proved especially helpful in the context of style transfer ([Ulyanov et al., 2017](#ulyanov17improved)).

<figure id="fig_norm">
    <img src="{{ site.url }}/public/images/2022-03-25-unnormalized-resnets/normalisation_dimensions.svg" alt="visualization of normalization methods">
    <figcaption>
        Figure&nbsp;2: Normalization methods (Batch, Layer, Instance and Group Normalization) and the parts of the input they compute their statistics over.
        $|\mathcal{B}|,$ $C,$ and $S$ are batch size, number of channels/features and signal size, respectively (cf. Figure&nbsp;<a href="#fig_dims">1</a>).
        The lightly shaded region for LN indicates the additional context that is typically used for image data.
        Image has been adapted from (<a href="#wu18groupnorm">Wu & He, 2018</a>).
    </figcaption>
</figure>

Instead of normalizing the inputs, it is also possible to get a normalizing effect by rescaling the weights of the network ([Arpit et al., 2016](#arpit16normprop)).
Especially in convolutional networks, this can significantly reduce the computational overhead.
With **Weight Normalization (WN)** ([Salimans & Kingma, 2016](#salimans16weightnorm)), the weight vectors for each neuron are normalized to have unit norm.
This idea can also be found in a(n independently developed) technique called **Normalization Propagation (NP)** ([Arpit et al., 2016](#arpit16normprop)).
However, in contrast to WN, NP accounts for the effect of (ReLU) activation functions.
In some sense, NP can be interpreted as a variant of BN where the statistics are computed theoretically (in expectation) rather than on the fly.
**Spectral Normalization (SN)**, on the other hand, makes use of an induced matrix norm to normalize the entire weight matrix ([Miyato et al., 2018](#miyato18spectralnorm)).
Concretely, the weights are scaled by the reciprocal of an approximation of the largest singular value of the weight matrix.

Whereas WN, NP and SN still involve the computation of some weight norm, it is also possible to obtain normalization without any computational overhead.
By creating a forward pass that induces attracting fixed points in mean and variance, **Self-Normalizing Networks (SNNs)** ([Klambauer et al., 2017](#klambauer17selfnorm)) are able to effectively normalize the signal.
To achieve these fixed points, it suffices to carefully scale the ELU activation function ([Clevert et al., 2016](#clevert16elu)) and the initial variance of the weights.
Additionally, [Klambauer et al. (2017)](#klambauer17selfnorm) provide a way to tweak dropout so that it does not interfere with the normalization.
Maybe it is useful to point out that SNNs do not consist of explicit normalization operations.
In this sense, an SNN could already be seen as an example of _normalizer-free_ networks.


## Skip Connections

With normalization out of the way, we probably want to tackle the _skip connections_.
After all, [Brock et al. (2021a)](#brock21characterizing) mainly aim to rid Residual Networks (ResNets) of normalization.
Although skip connections already existed long before ResNets were invented, they are often considered as one of the main contributions by the work of [He et al., 2016](#he16resnet).
In some sense, it almost seems as if skip connections could only become popular after BN was invented.
Especially if we consider the effects of skip connections on the statistics of signals flowing through the network.

### History

_Shortcut_ or _skip connections_ make it possible for information to bypass one or more layers in a neural network.
Mathematically, they are typically expressed using a formalism of the form

$$\boldsymbol{y} = \boldsymbol{x} + f(\boldsymbol{x}),$$

where $f$ represents some non-linear transformation ([He et al., 2016a](#he16resnet), [2016b](#he16preresnet)).
This non-linear transformation is typically a sub-network that is commonly referred to as the _residual branch_ or _residual connection_.
When the outputs of the residual branch have different dimensions, it is typical to use a linear transformation to match the output dimension of the skip connection with that of the residual connection.

Since it often helps to have a few lines of code to understand these vague descriptions, an implementation of the skip connections from ([He et al., 2016b](#he16preresnet)) is given below.
The comments aim to highlight the differences with the ResNets from ([He et al., 2016a](#he16resnet)).
For a complete implementation of this skip connection module, we refer to the [code](#pre-activation-resnets) at the end of this post.
```python
    def forward(self, x):
        x = self.preact(x)  # diff 1: compute global pre-activations
        skip = self.downsample(x)
        residual = self.residual_branch(x)
        # return torch.relu(residual + skip) (diff 2)
        return residual + skip
```

Skip connections became very popular in computer vision due to the work of He et al. ([2016a](#he16resnet)).
However, they were already commonly used as a trick to improve learning in multi-layer networks before deep learning was even a thing ([Ripley, 1996](#ripley96pattern)).
Similar to normalization methods, skip connections can improve the condition of the optimization problem by making it harder for the Hessian to become singular ([van der Smagt & Hirzinger, 1998](#vandersmagt98solving)).
However, skip connections also have benefits in the forward pass:
e.g., [Srivastava et al. (2015)](#srivastava15highway) argue that information should be able to flow through the network without being altered.
[He et al., (2016a)](#he16resnet), on the other hand, claim that learning should be easier if the network can focus on the non-linear part of the transformation (and ignore the linear component).

<figure id="fig_skip">
    <img src="{{ site.url }}/public/images/2022-03-25-unnormalized-resnets/skip_connections.svg" alt="visualization of different types of skip connections">
    <figcaption>
        Figure&nbsp;3: Variations on skip connections in ResNets, Densenets and Highway networks.
        The white blocks correspond to the input / skip connection and the blue blocks correspond to the output of the non-linear transformation.
        The greyscale blocks are values between zero and one and correspond to masks.
    </figcaption>
</figure>

The general formulation of skip connections that we provided earlier, captures the idea of skip connections very well.
As you might have expected, however, there are plenty of variations on the exact formulation (a few of which are illustrated in Figure&nbsp;[3](#fig_skip)).
Strictly speaking, even [He et al., (2016a)](#he16resnet) do not adhere to their own formulation because they apply an activation function on what we denoted as $\boldsymbol{y}$ ([He et al., 2016b](#he16preresnet); see code snippet).
In DenseNets ([G. Huang et al., 2017](#huang17densenet)), the outputs of the skip and residual connections are concatenated instead of aggregated by means of a sum.
This retains more of the information for subsequent layers.
Other variants of skip connections make use of masks to select which information is passed on.
Highway networks ([Srivasta et al., 2015](#srivasta15highway)) make use of a gating mechanism similar to that in Long Short-Term Memory (LSTM) ([Hochreiter et al., 1997](#hochreiter97lstm)).
These gates enable the network to learn how information from the skip connection is to be combined with that of the residual branch.
Similarly, Transformers ([Vaswani et al., 2017](#vaswani17attention)) could be interpreted as a variation on highway networks without residual branches.
This comparison does only hold, however, if you are willing to interpret the attention mask as some form of complex gate for the skip connection.

### Moment Control

Traditional initialization techniques manage to provide a stable starting point for the propagation of mean and variance in fully connected layers, but they do not work so well in ResNets.
The key problem is that the variance can not remain constant when simple additive skip connections are used.
After all, the variance is linear and unless the non-linear transformation branch would output a zero-variance signal, the output variance must be greater than the input variance.
Moreover, if the signal would have a strictly positive mean, also the mean would start drifting when residual layers are chained together.
Luckily, these drifting effects can be mitigated to some extent, e.g. by using BN.
However, are there alternative approaches and if yes, what are these approaches?

Before we come to possible solutions, it might be useful to point out that these drift effects are due to the simple _additive_ skip connections used in ResNets.
For example, the gating mechanism that is used to control the skip connection in highway networks makes the mean shift much less of a problem than in ResNets.
In the case of DenseNets, the concatenation does not affect either mean or variance if the residual branch produces outputs with similar statistics as the inputs.
Therefore, we mainly focus on these simple _additive_ skip connections in ResNets.

Similar to standard initialization methods, the key idea to counter drifting in ResNets is to stabilize the variance propagation.
To this end, a slightly modified formulation of skip connections is typically used (e.g., [Szegedy et al., 2016](#szegedy16inceptionv4); [Balduzzi et al., 2017](#balduzzi17shattered); [Hanin & Rolnick, 2018](#hanin18how)):

$$\boldsymbol{y} = \alpha \boldsymbol{x} + \beta f(\alpha \boldsymbol{x}),$$

which is equivalent to the original formulation when $\alpha = \beta = 1.$
The key advantage of this formulation is that the variance can be controlled (to some extent) by tuning the newly introduced scaling factors $\alpha$ and $\beta.$
In terms of code, these modifications could look something like

```python
    def forward(self, x):
        x = self.preact(self.alpha * x)
        skip = self.downsample(x)
        residual = self.residual_branch(x)
        return self.beta * residual + skip
```

A very simple counter-measure to the variance explosion in ResNets is to set $\alpha = 1 / \sqrt{2}$ ([Balduzzi et al., 2017](#balduzzi17shattered)).
Assuming that the residual branch approximately preserves the variance, the variances of $\boldsymbol{y}$ and $\boldsymbol{x}$ should be roughly the same.
In practice, however, it seems to be more common to tune the $\beta$ factor instead of $\alpha$ ([Balduzzi et al., 2017](#balduzzi17shattered)).
For instance, simply setting $\beta$ to some small value (e.g., in the range $[0.1, 0.3]$) can already help ResNets (with BN) to stabilize training ([Szegedy et al., 2016](#szegedy16inceptionv4)).
It turns out that having small values for $\beta$ can help to preserve correlations between gradients, which should benefit learning ([Balduzzi et al., 2017](#balduzzi17shattered)).

Similar findings were established through the analysis of the variance propagation in ResNets by [Hanin & Rolnick (2018)](#hanin18how).
Eventually, they propose to set $\beta = b^l$ after the $l$-th skip connection, with $0 < b < 1$ to make sure that the sum of scaling factors from all layers converges.
[Arpit et al. (2019)](#arpit19how) additionally take the backward pass into account and show that $\beta = L^{-1}$ provides stable variance propagation in a ResNet with $L$ skip connections.
Learning the scaling factor $\beta$ in each layer can also make it possible to keep the variance under control ([Zhang et al., 2019](#zhang19fixup); [De & Smith, 2020](#de20skipinit)).


## Normalizer-Free ResNets

It could be argued that the current popularity of skip connections is due to BN.
After all, without BN, the skip connections in ResNets would have suffered from the drifting effects discussed [earlier](#moment-control).
However, this does not take away that BN does have a few [practical issues](#alternatives) and there are alternative techniques to control these drifting effects.
Therefore, it makes sense to research the question of whether BN is just a useful or a _necessary_ component of the ResNet architecture.

### Old Ideas

Whereas some alternative normalization methods aim to simply provide normalization in scenarios where BN does not work so well, other methods have been explicitly designed to reduce or get rid of the normalization computations (e.g., [Arpit et al., 2016](#arpit16normprop); [Salimans & Kingma, 2016](#salimans16weightnorm); [Klambauer et al., 2017](#klambauer17selfnorm)).
Even the idea of training ResNets without BN is practically as old as ResNets themselves.
With their Layer-Sequential Unit-Variance (LSUV) initialization, [Mishkin et al. (2016)](#mishkin16lsuv) showed that it is possible to replace BN with good initialization for small datasets (CIFAR-10).
Similarly, [Arpit et al. (2019)](#arpit19) are able to close the gap between Weight Normalization (WN) and BN by reconsidering weight initialization in ResNets.

Getting rid of BN in ResNets was posed as an explicit goal by [Zhang et al. (2019)](#zhang19fixup), who proposed the so-called FixUp initialization scheme.
On top of introducing the learnable $\beta$ parameters and the $L^{-1/(2k - 2)}$ scaling for all layers $k$ in each of the $L$ residual branches,
they set the initial weights for the last layer in each residual branch to zero and introduce scalar biases before every layer in the network.
With these tricks, Zhang et al. show that FixUp can provide _almost_ the same benefits as BN for ResNets in terms of trainability and generalization.
Using a different derivation, [De & Smith (2020)](#de20skipinit) end up with a very similar solution to train ResNets without BN, which they term SkipInit.
The key difference with FixUp is that the initial value for the learnable $\beta$ parameter is set to be less than $1 / \sqrt{L}.$
As a result, SkipInit does not require the rescaling of initial weights in residual branches or setting weights to zero, which are considered crucial parts of the FixUp strategy ([Zhang et al. (2019)](#zhang19fixup)).

Also [Shao et al. (2020)](#shao20rescalenet) suggest to use a simple scaling strategy to replace BN in ResNets.
They propose to use a slightly modified scaling of the form, $\boldsymbol{y} = \alpha \boldsymbol{x} + \beta f(\boldsymbol{x}),$ where $\alpha^2 = 1 - \beta^2$ and $\beta^2 = 1 / (l + c)$ for the $l$-th skip connection.
Here, $c$ is an arbitrary constant, which was eventually set to be the number of residual branches, $L$.
For a single-layer ResNet ($l = c = 1$), this is equivalent to setting $\alpha = 1 / \sqrt{2},$ as suggested by [Balduzzi et al. (2017)](#balduzzi17shattered).
However, the more general approach should assure that the outputs of residual branches are weighted similarly at the output of the network, independent of their depth.

### Imitating Signal Propagation

Although the results of prior work look promising, there is still a performance gap compared to ResNets with BN.
To close this gap, [Brock et al. (2021a)](#brock21characterizing) suggest studying the propagation of mean and variance through ResNets by means of so-called Signal Propagation Plots (SPPs).
These SPPs simply visualize the squared mean and variance of the activations after each skip connection, as well as the variance at the end of every residual branch (before the skip connection).

To compute these values, the forward pass of the network must be slightly tweaked.
To this end, we can define a new method or a function that simulates the forward pass and extracts the necessary statistics for each skip connection, as follows:

```python
    @torch.no_grad()
    def signal_prop(self, x, dim=(0, -1, -2)):
        # forward code
        x = self.preact(x)
        skip = self.downsample(x)
        residual = self.residual_branch(x)
        out = residual + skip

        # compute necessary statistics
        out_mu2 = torch.mean(out.mean(dim) ** 2).item()
        out_var = torch.mean(out.var(dim)).item()
        res_var = torch.mean(residual.var(dim)).item()
        return out, (out_mu2, out_var, res_var)
```

This allows us to analyse the statistics for a single skip connection.
By propagating a white noise signal (e.g., `torch.randn(1000, 3, 224, 224))`) through the entire ResNet, we obtain the data that allows us to produce SPPs.
We refer to the end of this post for an example [implementation](#multi-layer-spp) of a full NF-ResNet with `signal_prop` method.

Figure&nbsp;[4](#fig_spp) provides an example of the SPPs for a pre-activation ResNets (or v2 ResNets, cf. [He et al., 2016b](#he16identity)) with and without BN.
The SPPs on the left clearly illustrate that BN transforms the exponential growth to a linear increase in ResNets, as described in theory (e.g., [Balduzzi et al., 2017](#balduzzi17shattered); [De & Smith, 2020](#de20skipinit)).
When focusing on ResNets with BN (on the right of Figure&nbsp;[4](#fig_spp)), it is clear that mean and variance are reduced after every sub-net, each of which consists of a few skip connections.
This reduction is due to the _pre-activation_ block (BN + ReLU) that is inserted between every two sub-nets in these ResNets (remember the code snippet from earlier?).

<figure id="fig_spp">
    <img src="{{ site.url }}/public/images/2022-03-25-unnormalized-resnets/spp.svg" alt="Image with two plots. The left plot shows two signal propagation plots: one for ResNets with (increasing gray lines) and one for ResNets without (approximately flat blue lines) Batch Normalization on a logarithmic scale. The right plot shows the zig-zag lines that represent the squared mean and variance after each residual branch." width="100%">
    <figcaption>
        Figure&nbsp;4: Example Signal Propagation Plots (SPPs) for a pre-activation (v2) ResNet-50 at initialization.
        SPPs plot the squared mean ($\mu^2$) and variance ($\sigma^2$) of the pre-activations after each skip connection ($x$-axis), as well as the variance of the residuals before the skip connection ($\sigma_f^2$, $y$-axis on the right).
        The left plot illustrates the difference between ResNets with and without BN layers.
        The plot on the right shows the same SPP for a ResNet with BN without the logarithmic scaling (cf. "<em>BN->ReLU</em>" in Figure&nbsp;1, <a href="#brock21characterizing">Brock et al., 2021a</a>).
        Note that ResNet-50 has four sub-nets with 3, 4, 6 and 3 skip connections, respectively.
    </figcaption>
</figure>

The goal of Normalizer-Free ResNets (NF-ResNets) is to get rid of the BN layers in ResNets while preserving the characteristics visualized in the SPPs ([Brock et al., 2021a](#brock21characterizing)).
To get rid of the exponential variance increase in unnormalized ResNets, it suffices to set $\alpha = 1 / \sqrt{\operatorname{Var}[\boldsymbol{x}]}$ in our modified formulation of ResNets.
Here, $\operatorname{Var}[\boldsymbol{x}]$ is the variance over all samples in the dataset, such that the $\alpha$ scaling effectively mirrors the division by $\boldsymbol{\sigma}_\mathcal{B}$ in BN (assuming a large enough batch size).
Unlike BN, however, the scaling in NF-ResNets is computed analytically for every skip connection.
This is possible if the inputs to the network are properly normalized (i.e., have unit variance) and if the residual branch, $f$, properly preserves variance (i.e. is initialized correctly).
The $\beta$ parameter, on the other hand, is simply used as a hyper-parameter to directly control the variance increase after every skip connection.

It might be useful to point out that the proposed $\alpha$ scaling does not perfectly conform with our general formulation for ResNets.
After all, the pre-activation layers mostly end up affecting only the inputs to the residual branch, such that $\boldsymbol{y} = \boldsymbol{x} + \beta f(\alpha \boldsymbol{x})$ (see [code](#extra-code-snippets) for details).
Only between the different sub-networks, which consist of multiple skip connections, the pre-activations are applied globally and the signal will be normalized.
This also explains the variance drops in the SPPs for regular ResNets (see Figure&nbsp;[4](#fig_spp)).
Note that this also means that the variance within sub-networks of an NF-ResNet will increase in the same way as for a ResNet with BN.
Although it would have been perfectly possible to maintain a steady variance, NF-ResNets are effectively designed to mimic the signal propagation due to BN layers in regular ResNets.

<figure id="fig_nfresnet">
    <img src="{{ site.url }}/public/images/2022-03-25-unnormalized-resnets/spp_nfresnet.svg" alt="Image with two plots. The left plot shows two SPPs: one for a ResNet with Batch Normalization (gray lines) and one for a Normalizer-Free ResNet (blue lines). The curves representting variance for both models are very close to each other, but the curve for the mean is quite different. The right plot is similar, but now the blue mean and residual variance curves are zero and one everywhere, respectively." width="100%">
    <figcaption>
        Figure&nbsp;5: SPPs comparing an NF-ResNet-50 to a Resnet with BN at initialization.
        The NF-ResNet in the left plot only uses the $\alpha$ and $\beta$ scaling parameters (cf. "<em>NF, He Init</em>" in Figure&nbsp;2, <a href="#brock21characterizing">Brock et al., 2021a</a>).
        The right plot displays the behavior of an NF-ResNet with Centered Weight Normalization (cf. "<em>NF, Scaled WS</em>" in Figure&nbsp;2, <a href="#brock21characterizing">Brock et al., 2021a</a>).
        Note that the variance of the residuals in the right plot should give some insights as to why the curves do not overlap.
    </figcaption>
</figure>

As can be seen on the left plot in Figure&nbsp;[5](#fig_nfresnet), a plain NF-ResNet effectively imitates the variance propagation of the baseline ResNet pretty accurately.
The propagation of the squared mean in NF-ResNets, on the other hand, looks nothing like that from the BN model.
After all, the considerations that lead to the scaling parameters only cover the variance propagation.
On top of that, it turns out that the variance of the residual branches (right before it is merged with the skip connection) is not particularly steady.
This indicates that the residual branches do not properly preserve variance, which is necessary for the analytic computations of $\alpha$ to be correct.

It turns out that both of these discrepancies can be resolved by introducing a variant of Centered Weight Normalization (CWN; [L. Huang et al., 2017](#huang17centred)) to NF-ResNets.
CWN simply applies WN after subtracting the weight mean from each weight vector, which ensures that every output has zero mean and that the variance of the weights is constant.
[Brock et al. (2021a)](#brock21characterizing) additionally rescale the normalized weights to account for the effect of activation functions (cf. [Arpit et al., 2016](#arpit16normprop)).
The effect of including the rescaled CWN in NF-ResNets is illustrated in the right part of Figure&nbsp;[5](#fig_nfresnet).

### Performance

Empirically, [Brock et al. (2021a)](#brock21characterizing) show that NF-ResNets with standard regularization methods perform on par with traditional ResNets that are using BN.
An important [detail](https://github.com/deepmind/deepmind-research/blob/ba761289c157fc151c7f06aa37b812d8100561db/nfnets/resnet.py#L158-L159) that is not apparent from the text, however, is that their baseline ResNets use the (standard) "_BN -> ReLU_" order and not the "_ReLU -> BN_" order, which served as the model for the signal propagation of NF-ResNets.
This is also why the SPPs in Figure&nbsp;[5](#fig_nfresnet), which depict the "_ReLU -> BN_" order, do not perfectly overlap, unlike the figures in ([Brock et al., 2021a](#borck21characterizing)).

Because BN does induce computational overhead, it seems natural to expect NF-ResNets to allow for more computationally efficient models.
Therefore, [Brock et al. (2021a)](#brock21characterizing) also compare NF-ResNets with a set of architectures that are optimized for efficiency.
However, it turns out that some of these architectures do not play well with the weight normalization that is typically used in NF-ResNets.
As a result, normalizer-free versions of EfficientNets ([Tan & Le, 2019](#tan19efficientnet)) lag behind their BN counterparts.
When applied to (naive) RegNets ([Radosavovic et al., 2020](#radosovic20regnet)), however, the performance gap between with EfficientNets can be reduced by introducing the NF-ResNet scheme.
In subsequent work, [Brock et al. (2021b)](#brock21highperformance) show that NF-ResNets in combination with gradient clipping are able to outperform similar networks with BN.


## Discussion

NF-ResNets show that it is possible to build networks without BN that are able to achieve competitive prediction performance.
It is not yet entirely clear whether the ideas of NF-ResNets could make BN entirely obsolete, however.
Therefore, it should be interesting to take a closer look at what the limitations of NF-ResNets are.
Assuming that the ideas in NF-ResNets can make BN (at least partly) obsolete, this should also provide some insights as to what the important factors are to explain the success of BN.

### Limitations

First of all, the exact procedure for scaling residual branches is only meaningful for architectures that make use of simple additive skip connections.
This means that it is not possible to directly apply the ideas behind NF-ResNets on arbitrary architectures to get rid of BN layers.
Even similar architectures that make use a different kind of skip connection (e.g., DenseNets, Highway Networks, ...) are probably not compatible with this exact approach.
Furthermore, NF-ResNets still rely on (other) normalization methods to attain good performance &mdash; in contrast to what their name might suggest.
[Brock et al. (2021a)](#brock21characterizing) emphasize that they effectively do away with _activation normalization_, but they do rely on an adaptation of Weight Normalization to replace BN.
In this sense, it is arguable whether NF-ResNets are truly normalizer-free.
Finally, some of the problems with BN are not resolved or reintroduced when building competitive NF-ResNets.
E.g., there are still differences between training and testing when using plain dropout regularization, CWN still introduces a certain computational overhead during training, etc.

### Insights

In the end, an NF-ResNet can be interpreted as consisting of different components that model parts of what BN normally does.
For example, the $\alpha$ scaling factor used in NF-ResNets clearly models the division by the standard deviation of BN.
It is also easy to see that the implicit regularization that is attributed to BN can be replaced by explicit regularization schemes.
Furthermore, the mean subtraction in BN is practically implemented by the weight centering in CWN.
Also, the scale-invariance of the weights due to BN is re-introduced through CWN.
However, the input scale-invariance that BN introduces in each layer is lost when using CWN.
When considering the entire residual branch (or network), however, $\alpha$ does enable some sort of scale-invariance for the entirety of this branch (or network).
Finally, the affine transformation after the normalization in BN is modeled by scaling the result of CWN.
Note that the affine shift does not need to be modeled explicitly, since CWN does not annihilate the regular bias parameters of the layers it acts upon, in contrast to BN.

Although the effects of BN on the forward pass seem to be modeled quite well by NF-ResNets, the effects on the backward pass seem to be largely ignored by [Brock et al. (2021a)](#brock21characterizing).
This might indicate that the performance differences might be explained by the effect of BN on the backward pass.
Follow-up work by [Brock et al. (2021b)](#brock21highperformance) also suggest that these effects might not be unimportant.
After all, the gradient flow in NF-ResNets is only affected by the scaling factors, $\alpha$ and $\beta,$ since CWN does not otherwise affect the gradients w.r.t. the inputs.
Therefore, regular NF-ResNets do not have a gradient centering ([Schraudolph, 1998](#schraudolph98centering)) component, as can be found in BN layers.
However, an adaptive gradient clipping scheme ([Brock et al. 2021](#brock21highperformance)) seems to provide an effective alternative to what BN does in the backward pass.

### Conclusion

NF-ResNets show that it is possible to get rid of BN in ResNets without throwing away predictive performance.
However, NF-ResNets still rely on weight normalization schemes to make the models competitive with their BN counterparts.
Therefore, it could be argued that NF-ResNets are not entirely _normalizer-free_.
It almost seems as if NF-ResNets are an example of how BN can be imitated using different components, rather than how to get rid of it.
This also means that it is hard to distil meaningful insights as to why/how BN works so well.
One thing that this approach does make clear is that the backward dynamics due to BN should be part of the explanation.

In terms of the questions we set out to answer at the start, we could summarize as follows:

 1. Why get rid of BN in the first place[?](#alternatives)
    <br/>The dependency on batch statistics does raise some concerns.
 2. How (easy is it) to get rid of BN in ResNets[?](#moment-control)
    <br/>Although it is one of the reasons why ResNets made skip connections so popular,
    there are plenty of alternative tricks that can achieve similar effects.
 3. Is BN going to become obsolete in the near future[?](#limitations)
    <br/>It does not look like BN will disappear soon, because the techniques to get rid of BN are probably too specific to the ResNet architecture.
 4. Does this allow us to gain insights into why BN works so well[?](#insights)
    <br/>NF-ResNets practically copy the forward dynamcis of ResNets with BN, which seems to suggest that the backward dynamics of BN play an important role.
 5. Wait a second... Are they getting rid of normalization or just BN[?](#conclusion)
    <br/>Despite their name, NF-ResNets merely replace BN by another normalization technique.

PS: The question marks link to the relevant sections in case you would like some more detail after all.

**TL;DR:** NF-ResNets, rescaled ResNets with Centered Weight Normalization, can be used to imitate the forward pass of ResNets with BN, but they do not help much to explain what makes BN so successful.

---

### Acknowledgements

Special thanks go to Katharina Prinz, [Niklas Schmidinger](https://www.niklasschmidinger.com/) and the anonymous reviewer for their constructive feedback.

The ELLIS Unit Linz, the LIT AI Lab, the Institute for Machine Learning, are supported by the Federal State Upper Austria.
IARAI is supported by Here Technologies. 
We thank the projects AI-MOTION (LIT-2018-6-YOU-212), AI-SNN (LIT-2018-6-YOU-214), DeepFlood (LIT-2019-8-YOU-213), Medical Cognitive Computing Center (MC3), INCONTROL-RL (FFG-881064), PRIMAL (FFG-873979), S3AI (FFG-872172), DL for GranularFlow (FFG-871302), AIRI FG 9-N (FWF-36284, FWF-36235), ELISE (H2020-ICT-2019-3 ID: 951847). 
We thank Audi.JKU Deep Learning Center, TGW LOGISTICS GROUP GMBH, Silicon Austria Labs (SAL), FILL Gesellschaft mbH, Anyline GmbH, Google, ZF Friedrichshafen AG, Robert Bosch GmbH, UCB Biopharma SRL, Merck Healthcare KGaA, Verbund AG, Software Competence Center Hagenberg GmbH, TÜV Austria, Frauscher Sensonic and the NVIDIA Corporation.

---

## Extra Code Snippets

To facilitate the implementation of pre-residual networks in pytorch and to give a full example of how to implement the signal propagation plotting, we provide additional code snippets in [PyTorch](https://pytorch.org).

#### Pre-activation ResNets

The first snippet implements skip connections according to ([He et al., 2016b](#he16preresnet)).
The comments aim to highlight the differences with the ResNets from ([He et al., 2016a](#he16resnet)), for which an [implementation](https://github.com/pytorch/vision/blob/v0.11.2/torchvision/models/resnet.py#L86-L141) is included in the [Torchvision](https://pytorch.org/vision/stable/models.html#id10) library.

```python
{% include 2022-03-25-unnormalized-resnets/preresnet_block.py %}
```

#### NF-ResNets

When comparing the code for a skip connection between an NF-ResNet and a regular batch-normalized ResNet, we find that there are only a few minor changes.
So much so that it is more efficient to consider the `diff` output than the full code.

```diff
{%include 2022-03-25-unnormalized-resnets/nfresnet_block.patch %}
```

The patch above shows that apart from removing the BN layers and introducing the $\alpha$ and $\beta$ parameters, the BN layer in the pre-activation has to be replaced by the $\alpha$ scaling that is introduced in NF-ResNets.
These changes are effectively everything that needs to be done.
To be fair, this `Scaling` module is not standard in PyTorch, but it is easy enough to create it:

```python
{% include 2022-03-25-unnormalized-resnets/scaling.py %}
```

Putting everything together, including the `signal_prop` method introduced [earlier](#imitating-signal-propagation), the resulting code should correspond to the following:

```python
{% include 2022-03-25-unnormalized-resnets/nfresnet_block.py %}
```

The code for a full NF-ResNet (with multiple multi-layer sub-nets) can be found in a code snippets for [multi-layer SPPs](#multi-layer-spps).

#### Multi-layer SPPs

In order to give an example of how to collect the SPP data for a multi-layer ResNet, the snippet below provides code for an NF-ResNet.
For the sake of _brevity_, the implementation for CWN has been omitted here.
This code is inspired by the [`ResNet`](https://github.com/pytorch/vision/blob/v0.11.2/torchvision/models/resnet.py#L144-L249) implementation from Torchvision.
If you want to use this code, make sure that the `NFResidualBottleneck` module also provides a `signal_prop` method, as introduced [earlier](#imitating-signal-propagation).

```python
{% include 2022-03-25-unnormalized-resnets/nfresnet_spp.py %}
```


## References

<span id="arpit16normprop">Arpit, D., Zhou, Y., Kota, B., & Govindaraju, V. (2016). Normalization Propagation: A Parametric Technique for Removing Internal Covariate Shift in Deep Networks. 
Proceedings of The 33rd International Conference on Machine Learning, 48, 1168–1176.</span> 
([link](https://proceedings.mlr.press/v48/arpitb16.html),
 [pdf](http://proceedings.mlr.press/v48/arpitb16.pdf))

<span id="arpit19how">Arpit, D., Campos, V., & Bengio, Y. (2019). How to Initialize your Network? Robust Initialization for WeightNorm & ResNets. 
Advances in Neural Information Processing Systems, 32, 10902–10911.</span>
([link](https://papers.nips.cc/paper/2019/hash/e520f70ac3930490458892665cda6620-Abstract.html),
 [pdf](https://papers.nips.cc/paper/2019/file/e520f70ac3930490458892665cda6620-Paper.pdf))

<span id="ba16layernorm">Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization [Preprint]. </span> 
([link](http://arxiv.org/abs/1607.06450),
 [pdf](http://arxiv.org/pdf/1607.06450.pdf))

<span id="balduzzi17shattered">Balduzzi, D., Frean, M., Leary, L., Lewis, J. P., Ma, K. W.-D., & McWilliams, B. (2017). The Shattered Gradients Problem: If resnets are the answer, then what is the question? 
Proceedings of the 34th International Conference on Machine Learning, 70, 342–350.</span> 
([link](https://proceedings.mlr.press/v70/balduzzi17b.html),
 [pdf](http://proceedings.mlr.press/v70/balduzzi17b/balduzzi17b.pdf))

<span id="bjorck18understanding">Bjorck, N., Gomes, C. P., Selman, B., & Weinberger, K. Q. (2018). Understanding Batch Normalization. 
Advances in Neural Information Processing Systems, 31, 7694–7705. </span> 
([link](https://proceedings.neurips.cc/paper/2018/hash/36072923bfc3cf47745d704feb489480-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2018/file/36072923bfc3cf47745d704feb489480-Paper.pdf))

<span id="brock21characterizing">Brock, A., De, S., & Smith, S. L. (2021a). Characterizing signal propagation to close the performance gap in unnormalized ResNets. 
International Conference on Learning Representations 9.</span>
([link](https://openreview.net/forum?id=IX3Nnir2omJ),
 [pdf](https://openreview.net/pdf?id=IX3Nnir2omJ))

<span id="brock21highperformance">Brock, A., De, S., Smith, S. L., & Simonyan, K. (2021b). High-Performance Large-Scale Image Recognition Without Normalization [Preprint].</span>
([link](http://arxiv.org/abs/2102.06171),
 [pdf](http://arxiv.org/pdf/2102.06171.pdf))

<span id="clevert16elu">Clevert, D.-A., Unterthiner, T., & Hochreiter, S. (2016). Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). 
International Conference on Learning Representations 4.</span> 
([link](http://arxiv.org/abs/1511.07289),
 [pdf](http://arxiv.org/pdf/1511.07289.pdf))

<span id="de20skipinit">De, S., & Smith, S. L. (2020). Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks. 
Advances in Neural Information Processing Systems, 33, 19964–19975.</span>
([link](https://proceedings.neurips.cc//paper/2020/hash/e6b738eca0e6792ba8a9cbcba6c1881d-Abstract.html),
 [pdf](https://proceedings.neurips.cc//paper/2020/file/e6b738eca0e6792ba8a9cbcba6c1881d-Paper.pdf))

<span id="gitman17comparison">Gitman, I., & Ginsburg, B. (2017). Comparison of Batch Normalization and Weight Normalization Algorithms for the Large-scale Image Classification [Preprint]. </span> 
g([link](http://arxiv.org/abs/1709.08145),
 [pdf](http://arxiv.org/pdf/1709.08145.pdf))

<span id="hanin18how">Hanin, B., & Rolnick, D. (2018). How to Start Training: The Effect of Initialization and Architecture. 
Advances in Neural Information Processing Systems, 31, 571–581.</span>
([link](https://proceedings.neurips.cc/paper/2018/hash/d81f9c1be2e08964bf9f24b15f0e4900-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2018/file/d81f9c1be2e08964bf9f24b15f0e4900-Paper.pdf))

<span id="he15delving">He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. 
Proceedings of the IEEE International Conference on Computer Vision, 1026–1034.</span> 
([link](https://doi.org/10.1109/ICCV.2015.123),
 [pdf](https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))

<span id="he16resnet">He, K., Zhang, X., Ren, S., & Sun, J. (2016a). Deep Residual Learning for Image Recognition. 
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.</span> 
([link](https://doi.org/10.1109/CVPR.2016.90),
 [pdf](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf))
 
<span id="he16preresnet">He, K., Zhang, X., Ren, S., & Sun, J. (2016b). Identity Mappings in Deep Residual Networks. 
In B. Leibe, J. Matas, N. Sebe, & M. Welling (Eds.), Computer Vision – ECCV 2016 (pp. 630–645). Springer International Publishing. </span> 
([link](https://doi.org/10.1007/978-3-319-46493-0_38),
 [pdf](https://arxiv.org/pdf/1603.05027.pdf))

<span id="hochreiter97lstm">Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. 
Neural Computation, 9(8), 1735–1780. </span> 
([link](https://doi.org/10.1162/neco.1997.9.8.1735),
 [pdf](https://ml.jku.at/publications/older/2604.pdf))

<span id="hoffer18norm">Hoffer, E., Banner, R., Golan, I., & Soudry, D. (2018). Norm matters: Efficient and accurate normalization schemes in deep networks. 
Advances in Neural Information Processing Systems, 31, 2160–2170. </span> 
([link](https://proceedings.neurips.cc/paper/2018/hash/a0160709701140704575d499c997b6ca-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2018/file/a0160709701140704575d499c997b6ca-Paper.pdf))

<span id="huang17centred">Huang, L., Liu, X., Liu, Y., Lang, B., & Tao, D. (2017). Centered Weight Normalization in Accelerating Training of Deep Neural Networks. 
Proceedings of the IEEE International Conference on Computer Vision, 2822–2830.</span> 
([link](https://doi.org/10.1109/ICCV.2017.305),
 [pdf](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Centered_Weight_Normalization_ICCV_2017_paper.pdf))

<span id="huang17densenet">Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. 
2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2261–2269. </span> 
([link](https://doi.org/10.1109/CVPR.2017.243),
 [pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf))

<span id="ioffe15batchnorm">Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. 
Proceedings of the 32nd International Conference on Machine Learning, 37, 448–456.</span> 
([link](http://proceedings.mlr.press/v37/ioffe15.html),
 [pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))

<span id="ioffe17batchrenorm">Ioffe, S. (2017). Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models. 
Advances in Neural Information Processing Systems, 30, 1945–1953. </span> 
([link](https://proceedings.neurips.cc/paper/2017/hash/c54e7837e0cd0ced286cb5995327d1ab-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2017/file/c54e7837e0cd0ced286cb5995327d1ab-Paper.pdf))

<span id="klambauer17selfnorm">Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-Normalizing Neural Networks. 
Advances in Neural Information Processing Systems, 30, 971–980.</span> 
([link](https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html),
 [pdf](https://papers.nips.cc/paper/2017/file/5d44ee6f2c3f71b73125876103c8f6c4-Paper.pdf))

<span id="lecun98efficient">LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). Efficient BackProp. 
In G. B. Orr & K.-R. Müller (Eds.), Neural Networks: Tricks of the Trade (1st ed., pp. 9–50). Springer. </span> 
([link](https://doi.org/10.1007/3-540-49430-8_2),
 [pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf))

<span id="li19understanding">Li, X., Chen, S., Hu, X., & Yang, J. (2019). Understanding the Disharmony Between Dropout and Batch Normalization by Variance Shift. 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2682–2690. </span> 
([link](https://doi.org/10.1109/CVPR.2019.00279),
 [pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf))

<span id="luo19towards">Luo, P., Wang, X., Shao, W., & Peng, Z. (2019). Towards Understanding Regularization in Batch Normalization. 6. </span>
([link](https://openreview.net/forum?id=HJlLKjR9FQ),
 [pdf](https://openreview.net/pdf?id=HJlLKjR9FQ))

<span id="miyato18spectralnorm">Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks. 
International Conference on Learning Representations 6.</span> 
([link](https://openreview.net/forum?id=B1QRgziT-),
 [pdf](https://openreview.net/pdf?id=B1QRgziT-))


<span id="mishkin16lsuv">Mishkin, D., & Matas, J. (2016). All you need is a good init. 
International Conference on Learning Representations 4.</span> 
([link](http://arxiv.org/abs/1511.06422),
 [pdf](http://arxiv.org/pdf/1511.06422.pdf))

<span id="radosavovic20regnet">Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollár, P. (2020). Designing Network Design Spaces. 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10425–10433.</span>
([link](https://doi.org/10.1109/CVPR42600.2020.01044),
 [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.pdf))

<span id="ripley96pattern">Ripley, B. D. (1996). Pattern Recognition and Neural Networks. Cambridge University Press. </span> 
([link](https://doi.org/10.1017/CBO9780511812651))

<span id="salimans16weightnorm">Salimans, T., & Kingma, D. P. (2016). Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. 
Advances in Neural Information Processing Systems, 29, 901–909.</span> 
([link](https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf))

<span id="santurkar18how">Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization? 
Advances in Neural Information Processing Systems, 31, 2483–2493.</span> 
([link](https://proceedings.neurips.cc/paper/2018/hash/905056c1ac1dad141560467e0a99e1cf-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2018/file/905056c1ac1dad141560467e0a99e1cf-Paper.pdf))

<span id="schraudolph98centering">Schraudolph, N. N. (1998). Centering Neural Network Gradient Factors. 
In G. B. Orr & K.-R. Müller (Eds.), Neural Networks: Tricks of the Trade (1st ed., pp. 207–226). Springer.</span> 
([link](https://doi.org/10.1007/3-540-49430-8_11),
 [pdf](https://n.schraudolph.org/pubs/Schraudolph98.pdf))

<span id="shao20rescalenet">Shao, J., Hu, K., Wang, C., Xue, X., & Raj, B. (2020). Is normalization indispensable for training deep neural network? 
Advances in Neural Information Processing Systems, 33, 13434–13444.</span>
([link](https://proceedings.neurips.cc/paper/2020/hash/9b8619251a19057cff70779273e95aa6-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2020/file/9b8619251a19057cff70779273e95aa6-Paper.pdf))

<span id="srivasta15highway">Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Training Very Deep Networks. 
Advances in Neural Information Processing Systems, 28, 2377–2385. </span> 
([link](https://papers.nips.cc/paper/2015/hash/215a71a12769b056c3c32e7299f1c5ed-Abstract.html), 
 [pdf](https://papers.nips.cc/paper/2015/file/215a71a12769b056c3c32e7299f1c5ed-Paper.pdf))

<span id="szegedy16inceptionv4">Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. (2016). Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning [Preprint].</span>
([link](http://arxiv.org/abs/1602.07261),
 [pdf](http://arxiv.org/pdf/1602.07261.pdf))

<span id="vandersmagt98solving">van der Smagt, P., & Hirzinger, G. (1998). Solving the Ill-Conditioning in Neural Network Learning. 
In G. B. Orr & K.-R. Müller (Eds.), Neural Networks: Tricks of the Trade (1st ed., pp. 193–206). Springer.</span> 
([link](https://doi.org/10.1007/3-540-49430-8_10))

<span id="vaswani17attention">Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. 
Advances in Neural Information Processing Systems, 30, 5998–6008.</span> 
([link](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf))


<span id="wadia21whitening">Wadia, N., Duckworth, D., Schoenholz, S. S., Dyer, E., & Sohl-Dickstein, J. (2021). Whitening and Second Order Optimization Both Make Information in the Dataset Unusable During Training, and Can Reduce or Prevent Generalization.
Proceedings of the 38th International Conference on Machine Learning, 139, 10617–10629.</span> 
([link](http://proceedings.mlr.press/v139/wadia21a.html),
 [pdf](http://proceedings.mlr.press/v139/wadia21a/wadia21a.pdf))

<span id="wu18groupnorm">Wu, Y., & He, K. (2018). Group Normalization. 
Computer Vision – ECCV 2018, 3–19. Springer International Publishing. </span> 
([link](https://doi.org/10.1007/978-3-030-01261-8_1),
 [pdf](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf))


<span id="zhang19fixup">Zhang, H., Dauphin, Y. N., & Ma, T. (2019). Fixup Initialization: Residual Learning Without Normalization. 
International Conference on Learning Representations 6. </span> 
([link](https://openreview.net/forum?id=H1gsz30cKX),
 [pdf](https://openreview.net/pdf?id=H1gsz30cKX))
