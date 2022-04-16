---
layout: post
title: Recent Advances in Deep Learning for Routing Problems
tags: [deep-learning, graph-neural-networks, combinatorial-optimization, travelling-salesperson-problem]
authors: Joshi, Chaitanya K.; Anand, Rishabh
---

**TL;DR** Developing neural network-driven solvers for combinatorial optimization problems such as the Travelling Salesperson Problem have seen a surge of academic interest recently. This blogpost presents a **Neural Combinatorial Optimization** pipeline that unifies several recently proposed model architectures and learning paradigms into one single framework. Through the lens of the pipeline, we analyze recent advances in deep learning for routing problems and provide new directions to stimulate future research towards practical impact.

{: class="table-of-content"}
* TOC
{:toc}

---

## Background on Combinatorial Optimization Problems

**Combinatorial Optimization** is a practical field in the intersection of mathematics and computer science that aims to solve constrained optimization problems which are NP-Hard. **NP-Hard problems** are challenging as exhaustively searching for their solutions is beyond the limits of modern computers. It is impossible to solve NP-Hard problems optimally at large scales. 

**Why should we care?** Because robust and reliable approximation algorithms to popular problems have immense practical applications and are the backbone of modern industries. For example, the **Travelling Salesperson Problem** (TSP) is the most popular Combinatorial Optimization Problems (COPs) and comes up in applications as diverse as logistics and scheduling to genomics and systems biology. 

> The Travelling Salesperson Problem is so famous, or notorious, that it even has an [xkcd comic](https://xkcd.com/399/) dedicated to it!

### TSP and Routing Problems

TSP is also a classic example of a **Routing Problem** -- Routing Problems are a class of COPs that require a sequence of nodes (e.g., cities) or edges (e.g., roads between cities) to be traversed in a specific order while fulfilling a set of constraints or optimising a set of variables. TSP requires a set of edges to be traversed in an order that ensures all nodes are visited exactly once. In the algorithmic sense, the optimal "tour" for our salesperson is a sequence of selected edges that provides the minimal distance or time taken over a Hamiltonian cycle, see Figure 1 for an illustration.

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/tsp-gif.gif" width="60%"/>
  <figcaption><b>Figure 1:</b> TSP asks the following question: Given a list of cities and the distances between each pair of cities, what is the <b>shortest possible route</b> that a salesperson can take to <b>visit each city</b> and <b>returns to the origin city</b>?
  (Source: <a href="http://mathgifs.blogspot.com/2014/03/the-traveling-salesman.html">MathGifs</a>)</figcaption>
</center></figure>

In real-world and practical scenarios, Routing Problems, or Vehicle Routing Problems (VRPs), can involve challenging constraints beyond the somewhat *vanilla* TSP; they are generalisations of TSP. For example, the **TSP with Time Windows** (TSPTW) adds a "time window" contraint to nodes in a TSP graph. This means certain nodes can either be active or inactive at a given time, i.e., they can only be visited during certain intervals. Another variant, the **Capacitated Vehicle Routing Problem** (CVRP) aims to find the optimal routes for a fleet of vehicles (i.e., multiple salespersons) visiting a set of customers (i.e., cities), with each vehicle having a maximum carrying capacity.

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/vrps.png" width="50%"/>
  <figcaption><b>Figure 2:</b> TSP and the associated class of Vehicle Routing Problems. VRPs can be characterized by their constraints, and this figure presents the relatively well-studied ones. There could be VRPs in the wild with <b>more complex</b> and <b>non-standard constraints</b>! (Source: adapted from <a href="https://ieeexplore.ieee.org/abstract/document/6887420">Benslimane and Benadada, 2014</a>)</figcaption>
</center></figure>

### Deep Learning to solve Routing Problems

Developing reliable algorithms and solvers for routing problems such as VRPs requires significant **expert intuition** and years of **trial-and-error**. For example, the state-of-the-art TSP solver, **Concorde**, leverages over 50 years of research on linear programming, cutting plane algorithms and branch-and-bound; here is an [inspiring video](https://www.youtube.com/watch?v=q8nQTNvCrjE) on its history. Concorde can find optimal solutions up to tens of thousands of nodes, but with extremely long execution time. As you can imagine, designing algorithms for complex VRPs is even more challegning and time consuming, especially with real-world constraints such as capacities or time windows in the mix.

This has led the machine learning community to ask the following question:

**Can we use deep learning to automate and augment expert intuition required for solving COPs?** 

> See this masterful survey from Mila for more in-depth motivation: [[Bengio et al., 2020](https://arxiv.org/abs/1811.06128)].

### Neural Combinatorial Optimization

[Neural Combinatorial Optimization](https://www.chaitjo.com/post/neural-combinatorial-optimization/) is an attempt to use **deep learning as a hammer** to hit the **COP nails**. Neural networks are trained to produce approximate solutions to COPs by directly learning from problem instances themselves. This line of research started at Google Brain with the seminal [Seq2seq Pointer Networks](https://arxiv.org/abs/1506.03134) and [Neural Combinatorial Optimization with RL](https://arxiv.org/abs/1611.09940) papers. Today, [Graph Neural Networks](https://arxiv.org/abs/2102.09544) are usually the architecture of choice at the core of deep learning-driven solvers as they tackle the graph structure of these problems.

Neural Combinatorial Optimization aims to improve over traditional COP solvers in the following ways:

- **No handcrafted heuristics.** Instead of application experts manually designing heuristics and rules, neural networks learn them via imitating an optimal solver or via reinforcement learning (we describe a pipeline for this in the next section).

- **Fast inference on GPUs.** Traditional solvers can often have prohibitive execution time for large-scale problems, e.g., Concorde took 7.5 months to solve the largest TSP with 109,399 nodes. On the other hand, once a neural network has been trained to approximately solve a COP, they have significantly favorable time complexity and can be parallelized via GPUs. This makes them highly desirable for real-time decision-making problems, especially routing problems.

- **Tackling novel and under-studied COPs.** The development of problem-specific COP solvers for novel or understudied problems that have esoteric constraints can be significantly sped up via neural combinatorial optimization. Such problems often arise in scientific discovery or computer architecture, e.g., an exciting success story is [Google's chip design system](https://www.nature.com/articles/s41586-021-03544-w) that will power the next generation of TPUs. You read that right -- **the next TPU chip for running neural networks has been designed by a neural network!**

---

## Unified Neural Combinatorial Optimization Pipeline

Using TSP as a canonical example, we now present a generic **neural combinatorial optimization pipeline** that can be used to characterize modern deep learning-driven approaches to several routing problems.

State-of-the-art approaches for TSP take the raw coordinates of cities as input and leverage **GNNs** or **Transformers** combined with classical **graph search** algorithms to constructively build approximate solutions. Architectures can be broadly classified as: (1) **autoregressive** approaches, which build solutions in a step-by-step fashion; and (2) **non-autoregressive** models, which produce the solution in one shot. Models can be trained to **imitate optimal solvers** via supervised learning or by minimizing the length of TSP tours via **reinforcement learning**.

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/pipeline-box.png" width="75%"/>
  <figcaption><b>Figure 3:</b> Neural combinatorial optimization pipeline (Source: <a href="https://arxiv.org/abs/2006.07054">Joshi et al., 2021</a>).</figcaption>
</center></figure>

The 5-stage pipeline from [Joshi et al., 2021](https://arxiv.org/abs/2006.07054) brings together prominent model architectures and learning paradigms into **one unified framework**. This will enable us to dissect and analyze recent developments in deep learning for routing problems, and provide new directions to stimulate future research.

### (1) Defining the problem via graphs

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/pipeline-1.png" width="60%"/>
  <figcaption><b>Figure 4: Problem Definition:</b> TSP is formulated via a fully-connected graph of cities/nodes, which can be sparsified further.</figcaption>
</center></figure>

TSP is formulated via a fully-connected graph where **nodes** correspond to **cities** and **edges** denote **roads** between them. The graph can be sparsified via heuristics such as k-nearest neighbors. This enables models to scale up to large instances where pairwise computation for all nodes is intractable [[Khalil et al., 2017](https://arxiv.org/abs/1704.01665)] or learn faster by reducing the search space [[Joshi et al., 2019](https://arxiv.org/abs/1906.01227)].

### (2) Obtaining latent embeddings for graph nodes and edges

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/pipeline-2.png" width="60%"/>
  <figcaption><b>Figure 5: Graph Embedding:</b> Embeddings for each graph node are obtained using a <b>Graph Neural Network</b> encoder, which builds local structural features via recursively aggregating features from each node's neighbors.</figcaption>
</center></figure>

A GNN or Transformer encoder computes **hiddden representations** or embeddings for each node and/or edge in the input TSP graph. At each layer, nodes gather features from their neighbors to represent **local graph structure** via recursive message passing. Stacking $L$ layers allows the network to build representations from the $L$-hop neighborhood of each node.

**Anisotropic** and **attention-based GNNs** such as Transformers [[Deudon et al., 2018](https://hanalog.polymtl.ca/wp-content/uploads/2018/11/cpaior-learning-heuristics-6.pdf), [Kool et al., 2019](https://arxiv.org/abs/1803.08475)] and Gated Graph ConvNets [[Joshi et al., 2019](https://arxiv.org/abs/1906.01227)] have emerged as the default choice for encoding routing problems. The attention mechanism during neighborhood aggregation is critical as it allows each node to weigh its neighbors based on their **relative importance** for solving the task at hand.

> Importantly, the Transformer encoder can be seen as an attentional GNN, i.e., [Graph Attention Network (GAT)](https://petar-v.com/GAT/), on a fully-connected graph. See [this blogpost](https://thegradient.pub/transformers-are-graph-neural-networks/) for an intuitive explanation.

### (3 + 4) Converting embeddings into discrete solutions

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/pipeline-3.png" width="70%"/>
  <figcaption><b>Figure 5: Solution Decoding and Search:</b> Probabilities are assigned to each node or edge for <b>belonging to the solution set</b> (here, an MLP makes a prediction per edge to obtain a 'heatmap' of edge probabilities), and then converted into <b>discrete decisions</b> through classical graph search techniques such as greedy search or beam search.</figcaption>
</center></figure>

Once the nodes and edges of the graph have been encoded into latent representations, we must decode them into discrete TSP solutions.
This is done via a two-step process: Firstly, probabilities are assigned to each node or edge for belonging to the solution set, either independent of one-another (i.e., **Non-autoregressive decoding**) or conditionally through graph traversal (i.e., **Autoregressive decoding**). Next, the predicted probabilities are converted into discrete decisions through classical **graph search techniques** such as greedy search or beam search guided by the probabilistic predictions (more on graph search later, when we discuss recent trends and future directions).

The choice of decoder comes with tradeoffs between **data-efficiency** and **efficiency of implementation**:
Autoregressive decoders [[Kool et al., 2019](https://arxiv.org/abs/1803.08475)] cast TSP as a Seq2Seq or **language translation task** from a set of unordered cities to an ordered tour. They explicitly model the **sequential inductive bias** of routing problems through step-by-step selection of one node at a time. On the other hand, Non-autoregressive decoders [[Joshi et al., 2019](https://arxiv.org/abs/1906.01227)] cast TSP as the task of producing **edge probability heatmaps**. The NAR approach is significantly faster and better suited for real-time inference as it produces predictions in **one shot** instead of step-by-step. However, it ignores the sequential nature of TSP, and may be less efficient to train when compared fairly to AR decoding [[Joshi et al., 2021](https://arxiv.org/abs/2006.07054)].

### (5) Training the model

Finally, the entire encoder-decoder model is trained in an **end-to-end** fashion, exactly like deep learning models for computer vision or natural language processing. In the simplest case, models can be trained to produce close-to-optimal solutions via **imitating an optimal solver**, i.e., via supervised learning. For TSP, the **Concrode** solver is used to generate labelled training datasets of optimal tours for millions of random instances. Models with AR decoders are trained via teacher-forcing to output the optimal sequence of tour nodes [[Vinyals et al., 2015](https://arxiv.org/abs/1506.03134)], while those with NAR decoders are trained to identify edges traversed during the tour from non-traversed edges [[Joshi et al., 2019](https://arxiv.org/abs/1906.01227)].

However, creating labelled datasets for supervised learning is an **expensive** and **time-consuming process**. Especially for very large problem instances, the exactness guarentees of optimal solvers may no longer materialise, leading to inexact solutions being used for supervised training. This is far from ideal from both practical and theoretical standpoints [[Yehuda et al., 2020](https://arxiv.org/abs/2002.09398)].

**Reinforcement learning** is a elegant alternative in the absence of groundtruth solutions, as is often the case for understudied problems. As routing problems generally require sequential decision making to **minimize a problem-specific cost functions** (e.g., the tour length for TSP), they can elegantly be cast in the RL framework which trains an agent to **maximize a reward** (the negative of the cost function). Models with AR decoders can be trained via standard policy gradient algorithms [[Kool et al., 2019](https://arxiv.org/abs/1803.08475)] or Q-Learning [[Khalil et al., 2017](https://arxiv.org/abs/1704.01665)].

---

## Characterizing Prominent Papers via the Pipeline

We can characterize prominent works in deep learning for TSP through the 5-stage pipeline. Recall that the pipeline consists of: (1) Problem Definition → (2) Graph Embedding → (3) Solution Decoding → (4) Solution Search → (5) Policy Learning. Starting from the Pointer Networks paper by Oriol Vinyals and collaborators, the following **table** highlights in <span style="color:red">Red</span> the major innovations and contributions for several notable and early papers.

| Paper | Definition | Graph Embedding | Solution Decoding | Solution Search | Policy Learning |
| --- | --- | --- | --- | --- | --- |
| [Vinyals et al., 2015](https://arxiv.org/abs/1506.03134) | Sequence | <span style="color:red">Seq2Seq</span> | <span style="color:red">Attention (AR)</span> | Beam Search | Immitation (SL) |
| [Bello et al., 2017](https://arxiv.org/abs/1611.09940) | Sequence | Seq2seq | Attention (AR) | Sampling | <span style="color:red">Actor-critic (RL)</span> |
| [Khalil et al., 2017](https://arxiv.org/abs/1704.01665) | <span style="color:red">Sparse Graph</span> | <span style="color:red">Structure2vec</span> | MLP (AR) | Greedy Search | <span style="color:red">DQN (RL)</span> |
| [Deudon et al., 2018](https://hanalog.polymtl.ca/wp-content/uploads/2018/11/cpaior-learning-heuristics-6.pdf) | Full Graph | <span style="color:red">Transformer Encoder</span> | Attention (AR) | Sampling + <span style="color:red">Local Search</span> | Actor-critic (RL) |
| [Kool et al., 2019](https://arxiv.org/abs/1803.08475) | Full Graph | <span style="color:red">Transformer Encoder</span> | Attention (AR) | Sampling | <span style="color:red">Rollout (RL)</span> |
| [Joshi et al., 2019](https://arxiv.org/abs/1906.01227) | Sparse Graph | <span style="color:red">Residual Gated GCN</span> | <span style="color:red">MLP Heatmap (NAR)</span> | Beam Search | Immitation (SL) | 
| [Ma et al., 2020](https://arxiv.org/abs/1911.04936) | Full Graph | GCN | <span style="color:red">RNN + Attention (AR)</span> | Sampling | Rollout (RL) |

---

## Recent Advances and Avenues for Future Work

With the unified 5-stage pipeline in place, let us highlight some **recent advances** and **trends** in deep learning for routing problems. We will also provide some future research directions with a focus on improving generalization to large-scale and real-world instances.

### Leveraging Equivariance and Symmetries

One of the most influential early works, the autoregressive Attention Model [[Kool et al., 2019](https://arxiv.org/abs/1803.08475)], considers TSP as a Seq2Seq language translation problem and sequentially constructs TSP tours as permutations of cities. One immediate drawback of this formulation is that it does not consider the **underlying symmetries of routing problems**.

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/pomo.png" width="75%"/>
  <figcaption><b>Figure 6:</b> In general, a TSP has one unique optimal solution (L). However, under the autoregressive formulation when a solution is represented as a sequence of nodes, <b>multiple optimal permutations</b> exist (R). (Source: <a href="https://arxiv.org/abs/2010.16011">Kwon et al., 2020</a>)</figcaption>
</center></figure>

**POMO: Policy Optimization with Multiple Optima** [[Kwon et al., 2020](https://arxiv.org/abs/2010.16011)] proposes to leverage invariance to the starting city in the constructive autoregressive formulation. They train the same Attention Model, but with a new reinforcement learning algorithm (step 5 in the pipeline) which exploits the existence of multiple optimal tour permutations. 

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/equivariance.png" width="75%"/>
  <figcaption><b>Figure 7:</b> TSP solutions remain unchanged under the <b>Euclidean symmtery group</b> of rotations, reflections, and translations to the city coordinates. Incorporating these symetries into the model may be a principled approach to tackling large-scale TSPs.</figcaption>
</center></figure>

Similarly, a very recent ugrade of the Attention model by [Ouyang et al., 2021](https://arxiv.org/abs/2110.03595) considers invariance with respect to **rotations, reflections,** and **translations** (i.e., the Euclidean symmetry group) of the input city coordinates. They propose an autoregressive approach while ensuring invariance by performing data augmentation during the problem definition stage (pipeline step 1) and using relative coordinates during graph encoding (pipeline step 2). Their approach shows particularly strong results on zero-shot generalization from random instances to the real-world TSPLib benchmark suite.

Future work may follow the [**Geometric Deep Learning (GDL)**](https://geometricdeeplearning.com/) blueprint for architecture design. GDL tells us to explicitly think about and incorporate the symmetries and inductive biases that govern the data or problem at hand. As routing problems are **embedded in euclidean coordinates** and the **routes are cyclical**, incorporating these contraints directly into the model architectures or learning paradigms may be a principled approach to improving generalization to large-scale instances greater than those seen during training.

### Improved Graph Search Algorithms

Another influential research direction has been the one-shot non-autoregressive Graph ConvNet approach [[Joshi et al., 2019](https://arxiv.org/abs/1906.01227)]. Several recent papers have proposed to retain the same Gated GCN encoder (pipeline step 2) while replacing the beam search component (pipeline step 4) with **more powerful** and **flexible graph search algorithms**, e.g., Dynamic Programming [[Kool et al., 2021](https://arxiv.org/abs/2102.11756)] or Monte-Carlo Tree Search (MCTS) [[Fu et al., 2020](https://arxiv.org/abs/2012.10658)].

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/heatmaps.png" width="75%"/>
  <figcaption><b>Figure 8:</b> The Gated GCN encoder <a href="https://arxiv.org/abs/1906.01227">[Joshi et al., 2019]</a> can be used to produce <b>edge prediction 'heatmaps'</b> (in transparent red color) for TSP, CVRP, and TSPTW. These can be further processed by <a href="https://arxiv.org/abs/2102.11756">DP</a> or <a href="https://arxiv.org/abs/2012.10658">MCTS</a> to output routes (in solid colors). The GCN essentially reduces the solution search space for sophisticated search algorithms which may have been intractable when searching over all possible routes. (Source: <a href="https://arxiv.org/abs/2102.11756">Kool et al., 2021</a>)</figcaption>
</center></figure>

The [GCN + MCTS framework](https://arxiv.org/abs/2012.10658) by Fu et al. in particular has a very interesting approach to **training models efficiently on trivially small TSP** and successfully **transferring the learnt policy to larger graphs** in a zero-shot fashion (something that the original GCN + Beam Search by Joshi et al. struggled with). They ensure that the predictions of the GCN encoder generalize from small to large TSP by updating the problem definition (pipeline step 1): large problem instances are represented as many smaller sub-graphs which are of the same size as the training graphs for the GCN, and then merge the GCN edge predictions before performing MCTS. 

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/sample-merge.png" width="70%"/>
  <figcaption><b>Figure 9:</b> The GCN + MCTS framework <a href="https://arxiv.org/abs/2012.10658">[Fu et al., 2020]</a> represents large TSPs as a set of <b>small sub-graphs</b> which are of the same size as the graphs used for training the GCN. Sub-graph edge heatmaps predicted by the GCN are merged together to obtain the heatmap for the full graph. This <b>divide-and-conquer approach</b> ensures that the embeddings and predictions made by the GCN generalize well from smaller to larger instances. (Source: <a href="https://arxiv.org/abs/2012.10658">Fu et al., 2020</a>)</figcaption>
</center></figure>

Originally proposed by [Nowak et al., 2018](https://openreview.net/forum?id=B1jscMbAW), this <b>divide-and-conquer strategy</b> ensures that the embeddings and predictions made by GNNs generalize well from smaller to larger TSP instances up to 10,000 nodes. Fusing GNNs, divide-and-conquer, and search strategies has similarly shown promising results for tackling large-scale CVPRs up to 3000 nodes [[Li et al., 2021](https://arxiv.org/abs/2107.04139)].

Overall, this line of work suggests that **stronger coupling** between the design of both the **neural** and **symbolic/search** components of models is essential for out-of-distribution generalization [[Lamb et al., 2020](https://arxiv.org/abs/2003.00330)]. However, it is also worth noting that designing highly customized and parallelized implementations of graph search on GPUs may be challenging for each new problem.

### Learning to Improve Sub-optimal Solutions

Recently, a number of papers have explored an alternative to constructive AR and NAR decoding schemes which involves **learning to iteratively improve (sub-optimal) solutions** or **learning to perform local search**, starting with [Chen et al., 2019](https://arxiv.org/abs/1810.00337) and [Wu et al., 2021](https://arxiv.org/abs/1912.05784). Other notable papers include the works of [Cappart et al., 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16484), [da Costa et al., 2020](https://arxiv.org/abs/2004.01608), [Ma et al., 2021](https://arxiv.org/abs/2110.02544), [Xin et al., 2021](https://arxiv.org/abs/2110.07983), and [Hudson et al., 2021](https://arxiv.org/abs/2110.05291).

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/cyclic-pe.png" width="75%"/>
  <figcaption><b>Figure 10:</b> Architectures which learn to improve sub-optimal TSP solutions by guiding decisions within local search algorithms. (a) The original Transformer encoder-decoder architecture <a href="https://arxiv.org/abs/1912.05784">[Wu et al., 2021]</a> which used <b>sinusoidal positional encodings</b> to represent the current sub-optimal tour permutation; (b) <a href="https://arxiv.org/abs/2110.02544">Ma et al., 2021</a>'s upgrade through the lens of symmetry: the Dual-aspect Transformer encoder-decoder  with <b>learnable positional encodings</b> which capture the cyclic nature of TSP tours; (c) Visualizations of sinusoidal vs. cyclical positional encodings.</figcaption>
</center></figure>

In all these works, since deep learning is used to **guide decisions** within classical local search algorithms (which are designed to work regardless of problem scale), this approach implicitly leads to **better zero-shot generalization** to larger problem instances compared to the constructive approaches. This is a very desirable property for practical implementations, as it may be intractable to train on very large or real-world TSP instances. 

Notably, **NeuroLKH** [[Xin et al., 2021](https://arxiv.org/abs/2110.07983)] uses edge probability heatmaps produced via GNNs to improve the **classical Lin-Kernighan-Helsgaun algorithm** and demonstrates strong zero-shot generalization to TSP with 5000 nodes as well as across TSPLib instances.

> For the interested reader, DeepMind's [Neural Algorithmic Reasoning](https://arxiv.org/abs/2105.02761) research program offers a unique meta-perspective on the intersection of neural networks with classical algorithms.

A limitation of this line of work is the prior need for **hand-designed local search algorithms**, which may be missing for novel or understudied problems. On the other hand, constructive approaches are arguably easier to adapt to new problems by enforcing constraints during the solution decoding and search procedure.

### Learning Paradigms that Promote Generalization

Future work could look at **novel learning paradigms** (pipeline step 5) which explicitly focus on generalization beyond supervised and reinforcement learning, e.g., [Hottung et al., 2020](https://openreview.net/forum?id=90JprVrJBO) explored autoencoder objectives to learn a continuous space of routing problem solutions.

At present, most papers propose to train models efficiently on trivially small and random TSPs, then transfer the learnt policy to larger graphs and real-world instances in a **zero-shot** fashion. The logical next step is to fine-tune the model on a small number of specifc problem instances. [Hottung et al., 2021](https://arxiv.org/abs/2106.05126) take a first step towards this by proposing to finetune a subset of model paramters for each specific problem instance via active search. In future work, it may be interesting to explore **fine-tuning as a meta-learning problem**, wherein the goal is to train model parameters specifically for fast adaptation to new data distributions and problems.

Another interesting direction could explore **tackling understudied routing problems** with challenging constraints via multi-task pre-training on popular routing problems such as TSP and CVPR, followed by problem-specific finetuning. Similar to **language modelling as a pre-training objective** in [Natural Language Processing](https://ruder.io/nlp-imagenet/), the goal of pre-training for routing would be to learn generally useful latent representations that can transfer well to novel routing problems.

### Improved Evaluation Protocols

Beyond algorithmic innovations, there have been repeated calls from the community for **more realistic evaluation protocols** which can lead to advances on real-world routing problems and adoption by industry [[Francois et al., 2019](https://arxiv.org/abs/1909.13121), [Yehuda et al., 2020](https://arxiv.org/abs/2002.09398)]. Most recently, [Accorsi et al., 2021](https://arxiv.org/abs/2109.13983) have provided an authoritative set of **guidelines for experiment design** and **comparisons** to classical Operations Research (OR) techniques. They hope that fair and rigorous comparisons on **standardized benchmarks** will be the first step towards the integration of deep learning techniques into industrial routing solvers.

In general, it is encouraging to see recent papers move beyond showing minor performance boosts on **trivially small random TSP instances**, and towards **embracing real-world benchmarks** such as [TSPLib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and [CVPRLib](http://vrp.atd-lab.inf.puc-rio.br/index.php/en/). Such routing problem collections contain graphs from cities and road networks around the globe along with their exact solutions, and have become the standard testbed for new solvers in the OR community. 

At the same time, we must be vary to not 'overfit' on the top `n` TSPLib or CVPRLib instances that every other paper is using. Thus, better synthetic datasets go hand-in-hand for benchmarking progress fairly, e.g., [Queiroga et al., 2021](https://openreview.net/forum?id=yHiMXKN6nTl) recently proposed a new libarary of synthetic 10,000 CVPR testing instances. Additionally, one can assess the robustness of neural solvers to small perturbations of problem instances with adversarial attacks, as proposed by [Geisler et al., 2021](https://arxiv.org/abs/2110.10942).

<figure><center>
  <img src="{{ site.url }}/public/images/2022-03-25-deep-learning-for-routing-problems/ml4co.png" width="25%"/>
  <figcaption><b>Figure 11:</b> Community contests such as <a href="https://www.ecole.ai/2021/ml4co-competition/">ML4CO</a> are a great initiative to track progress. (Source: ML4CO website).</figcaption>
</center></figure>

**Regular competitions** on freshly curated real-world datasets, such as the [ML4CO competition at NeurIPS 2021](https://arxiv.org/abs/2203.02433) and [AI4TSP at IJCAI 2021](https://arxiv.org/abs/2201.10453), are another great initiative to track progress in the intersection of deep learning and routing problems.

> We highly recommend the engaging panel discussion and talks from ML4CO, NeurIPS 2021, available on [YouTube](https://youtube.com/playlist?list=PLYWmzh0Y6EOZz3PtMxfaqEnRsfW-TF4nf).

---

## Summary

This blogpost presents a **neural combinatorial optimization pipeline** that unifies recent papers on deep learning for routing problems into a single framework. Through the lens of our framework, we then analyze and dissect recent advances, and speculate on directions for future research. 

The following table highlights in <span style="color:red">Red</span> the major innovations and contributions for recent papers covered in the previous sections.

| Paper | Definition | Graph Embedding | Solution Decoding | Solution Search | Policy Learning |
| --- | --- | --- | --- | --- | --- |
| [Kwon et al., 2020](https://arxiv.org/abs/2010.16011) | Full Graph | Transformer Encoder | Attention (AR) | Sampling | <span style="color:red">POMO Rollout (RL)</span> |
| [Fu et al., 2020](https://arxiv.org/abs/2012.10658) | <span style="color:red">Sparse Sub-graphs</span> | Residual Gated GCN | MLP Heatmap (NAR) | <span style="color:red">MCTS</span> | Immitation (SL) |
| [Kool et al., 2021](https://arxiv.org/abs/2102.11756) | Sparse Graph | Residual Gated GCN | MLP Heatmap (NAR) | <span style="color:red">Dynamic Programming</span> | Immitation (SL) |
| [Ouyang et al., 2021](https://arxiv.org/abs/2110.03595) | Full Graph + <span style="color:red">Data Augmentation</span> | <span style="color:red">Equivariant GNN</span> | Attention (AR) | Sampling + Local Search | <span style="color:red">Policy Rollout (RL)</span> |
| [Wu et al., 2021](https://arxiv.org/abs/1912.05784) | Sequence + <span style="color:red">Position</span> | Transformer Encoder | <span style="color:red">Transformer Decoder (L2I)</span> | Local Search | Actor-critic (RL) |
| [da Costa et al., 2020](https://arxiv.org/abs/2004.01608) | Sequence | GCN | <span style="color:red">RNN + Attention (L2I)</span> | Local Search | Actor-critic (RL) |
| [Ma et al., 2021](https://arxiv.org/abs/2110.02544) | Sequence + <span style="color:red">Cyclic Position</span> | <span style="color:red">Dual Transformer Encoder</span> | <span style="color:red">Dual Transformer Decoder (L2I)</span> | Local Search | <span style="color:red">PPO + Curriculum (RL)</span> |
| [Xin et al., 2021](https://arxiv.org/abs/2110.07983) | Sparse Graph | GAT | MLP Heatmap (NAR) | <span style="color:red">LKH Algorithm</span> | Immitation (SL) |
| [Hudson et al., 2021](https://arxiv.org/abs/2110.05291) | <span style="color:red">Sparse Dual Graph</span> | GAT | MLP Heatmap (NAR) | <span style="color:red">Guided Local Search</span> | Immitation (SL) |

---

As a final note, we would like to say that the **more profound motivation** of neural combinatorial optimization may NOT be to outperform classical approaches on well-studied routing problems. Neural networks may be used as a general tool for **tackling previously un-encountered NP-hard problems**, especially those that are non-trivial to design heuristics for. We are excited about recent applications of neural combinatorial optimization for [designing computer chips](https://www.nature.com/articles/s41586-021-03544-w), [optimizing communication networks](https://arxiv.org/abs/2109.10883), and [genome reconstruction](https://openreview.net/forum?id=1QxveKM654), and are looking forward to more in the future!

---

**Acknowledgements**: We would like to thank Goh Yong Liang, Yongdae Kwon, Yining Ma, Zhiguang Cao, Quentin Cappart, and Simon Geisler for helpful feedback and discussions.
