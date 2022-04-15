---
layout: post
title: Learning to Coarsen Graphs with Graph Neural Networks
tags: [relational learning, graph convolutions, graph theory]
authors: Suri, Karush
---


<p>With the rise of large-scale graphs for relational learning, graph coarsening emerges as a computationally viable alternative. We revisit the <cite><a href="https://openreview.net/pdf?id=uxpzitPEooJ">principles</a></cite> that aim to improve data-driven graph coarsening with adjustable coarsened structures.</p>

# Graph Coarsening 101  

> <p><b>TLDR</b>- Coarsening methods group nearby nodes to a supernode while preserving their relational properties. Traditional approaches only aim at node assignment with limited weight adjustment and similarity assessment.</p>

<p>Graph coarsening relates to the process of preserving node properties of a graph by grouping them into similarity clusters. These similarity clusters form the new nodes of the coarsened graph and are hence termed as <em>supernodes</em>. Contrary to <span class="popup" onclick="myFunction('myPopup')">partitioning methods <span class="popuptext" id="myPopup">graph partitioning is the process of segregating a graph into its sub-graphs.</span></span> which aim to extract information of local neighborhoods, coarsening aims to extract global representations of a graph. This implies that the coarsened graph must have all the node properties of the original graph preserved up to a certain level of accuracy.</p>

<p>Modern coarsening approaches aim to retain the spectral properties between original and coarse graphs. This involves the usage of <cite><a href="https://arxiv.org/pdf/1802.07510.pdf">spectral approximations</a></cite> and <cite><a href="https://proceedings.neurips.cc/paper/2019/file/cd474f6341aeffd65f93084d0dae3453-Paper.pdf">probabilistic frameworks</a></cite> to link nodes between the two structures. Although helpful, learning node assignments often results in lost edge information within nodes. Edge weights, during coarsening, are held fixed proportional to the total weights of incident edges between two given nodes. This hinders one to learn the connectivity of coarsened graphs.</p>

<p>In addition to connectivity, a spectral analysis of the coarsened graph is difficult to obtain. Assessing properties of different graph structures requires information about each node. Coarse graph supernodes, however, are themselves compositions of high-granularity nodes. This prevents a direct comparison between original and coarse representations. Let's formalize this problem in detail.</p>

<div class="center">
<div class="flip-card" style="text-align:center;margin-right:15px">
  <div class="flip-card-inner" style="text-align:center">
    <div class="flip-card-front" style="text-align:center">
      <img src="{{ site.url }}/public/images/2022-03-25-coarsening/graph_orig.PNG" alt="Avatar" style="width:200px;height:200px;border-radius: 20px;box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.19);">
    </div>
    <div class="flip-card-back" style="text-align:center;background-color: rgb(217,231,253);">
      <p>$$G = (V,E)$$</p>
      <p>$$\pi : V \rightarrow \hat{V}$$</p>
      <p>$$L = D - W$$</p>
    </div>
  </div>
</div>

<div class="flip-card" style="text-align:center;margin-left:15px;margin-right:15px">
  <div class="flip-card-inner" style="text-align:center">
    <div class="flip-card-front" style="text-align:center">
      <img src="{{ site.url }}/public/images/2022-03-25-coarsening/coarsen.PNG" alt="Avatar" style="width:200px;height:200px;border-radius: 20px;box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.19);">
    </div>
    <div class="flip-card-back" style="text-align:center;background-color: rgb(255,255,189);">
      <p>assign nodes: <span style="color:steelblue;">$\checkmark$</span></p>
      <p>compare structure: <span style="color:red;">$\times$</span></p>
      <p>adjust weights: <span style="color:red;">$\times$</span></p>
    </div>
  </div>
</div>

<div class="flip-card" style="text-align:center;margin-left:15px">
  <div class="flip-card-inner" style="text-align:center">
    <div class="flip-card-front" style="text-align:center">
      <img src="{{ site.url }}/public/images/2022-03-25-coarsening/graph_coarsen.PNG" alt="Avatar" style="width:200px;height:200px;border-radius: 20px;box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.19);">
    </div>
    <div class="flip-card-back" style="text-align:center;background-color: rgb(208,185,255);">
      <p>$$\hat{G} = (\hat{V},\hat{E})$$</p>
      <p>$$\pi^{-1} : \hat{V} \rightarrow V$$</p>
      <p>$$\hat{L} = \hat{D} - \hat{W}$$</p>
    </div>
  </div>
</div>
</div>
<p style="text-align:center;color:silver;font-size:15px;margin-top:10px;">The graph coarsening process (flip cards for details).</p>

<p>An original graph <span style="color:steelblue;">$G = (V,E)$</span> has <span style="color:steelblue;">$N = |V|$</span> nodes and standard <span class="popup" onclick="myFunction('newPopup')">Laplace operator <span class="popuptext" id="newPopup">Laplacians summarize spectral properties of graphs with symmetric and positive semi-definite structure.</span></span> <span style="color:steelblue;">$L = D-W$</span> expressing the difference between the degree matrix <span style="color:steelblue;">$D$</span> and edge weight matrix <span style="color:steelblue;">$W$</span>. The coarsened graph <span style="color:#7F00FF;">$\hat{G} = (\hat{V},\hat{E})$</span> has <span style="color:#7F00FF;">$n = |\hat{V}|$</span> nodes ($n < N$) and the Laplacian <span style="color:#7F00FF;">$\hat{L} = \hat{D}-\hat{W}$</span>. $G$ and $\hat{G}$ are linked via the vertex map $\pi: V \rightarrow \hat{V}$ such that all nodes $\pi^{-1}(\hat{v}) \in V$ are mapped to the supernode $\hat{v}$. The vertex map is thus a surjective mapping from the original graph to coarse graph. Due to a diference in the number of nodes, the two graphs denote different structures and hence, the Laplacians $L$ and $\hat{L}$ are not directy comparable.</p>

# Operations to Coarsen Graphs  

> <p><b>TLDR</b>- Assessment of Laplacian leads to a lift & project mapping formulation. In what follows, the formulation constructs a similarity measure $\mathcal{L}$ based on edge and vertex weights.</p>

<p>Establishing a link between original and coarsened graph requires a brief visit into their operators. To compare two graphs the paper defines operators on both original and coarse graphs. Let's begin by revisiting these operators for both structures.</p>

### Laplace This Way

<p>Since $G$ and $\hat{G}$ have different number of nodes, their Laplace operators $L$ and $\hat{L}$ are not directly comparable. Instead, we can utilize a functional mapping $\mathcal{F}$ which would remain invariant to the vertices of $G$ and $\hat{G}$. Such a mapping is intrinsic to the graph of interest, i.e.- the mapping would operate on $N$ vectors $f$ as well as $n$ vectors $\hat{f}$. This allows us to compare between <span style="color:steelblue;">$\mathcal{F}(L,f)$</span> and <span style="color:#7F00FF;">$\mathcal{F}(\hat{L},\hat{f})$</span>.</p>

<p>For the vertex map $\pi$, we set an $n \times N$ matrix P as,</p>

$$\begin{equation}
  P[r,i] =     
  \begin{cases}
      \frac{1}{|\pi^{-1}(\hat{v}_{r})|}, & \text{if}\ v_{i} \in \pi^{-1}(\hat{v}_{r}) \\
      0, & \text{otherwise}
    \end{cases}
\end{equation}$$


<p>for any $r \in [1,n]$. It is worth noting that the map $\pi$ may not always be given to us and can be approximated using a learning process. Next, we denote $P^{+}$ as the $N \times n$ <span class="popup" onclick="myFunction('anothPopup')">pseudo-inverse <span class="popuptext" id="anothPopup">Formally known as the Moore-Penrose inverse, the pseudo-inverse is an approximate inverse to matrices which may not invertible.</span></span> of P. Note that $P^{+}$ operates on the set $\hat{V}$ to provide a mapping in $V$. This allows us to formulate an operator on the coarsened set $\hat{V}$,</p>

<p style="text-align:center;color:#7F00FF;">
$$\begin{equation}
  \tilde{L} = (P^{+})^{T}LP^{+}
\end{equation}$$</p>

<p>$\tilde{L}$ operates on $n$ vectors $\hat{f} \in \mathbb{R}^{n}$ corresponding to the coarsened set. We look deeper into this mapping next.</p>


### Lift & Project

<p>We can intuitively understand the mapping $\tilde{L}$ as a lift and project mapping. Let's break this computation step-by-step. For a $n$ vector $\hat{f}$, the first operation $f = P^{+}\hat{f}$ lifts the $n$ dimensional representation to an $N$ dimensional one. The operation of $L$ then yields the desired result for spectral comparison. Following this computation, $Lf$ is again projected down to the $n$ vector space by the operation of $(P^{+})^{T}$.</p>

<p style="text-align:center;"><img src="{{ site.url }}/public/images/2022-03-25-coarsening/lift_project.PNG" style="width:640px;height:480px;display:inline-block;"></p>

<p style="text-align:center;color:silver;font-size:15px;margin-top:10px;">Lift and Project mappings between $n$ and $N$ vectors.</p>

Simultaneous actions of lift and project mappings give rise to a general framework between original and coarsened objects of the graph. The $\tilde{L}$ formulation for the above construction resembles the quadratic form $Q_{A}(x) = x^{T}Ax$ with $x^{T}$ as the lift $P^{+}$, $A$ as the Laplacian $L$ and $x$ as the projection map $P$. Similarly, an alternate construction could be obtained by considering the <a href="https://en.wikipedia.org/wiki/Rayleigh_quotient">Rayleigh Quotient</a> form,

$$\begin{equation}
  R_{A}(x) = \frac{x^{T}Ax}{x^{T}x}
\end{equation}$$

<p><em>Why the Rayleigh formulation?</em> This is because the eigenvectors and eigenvalues of the linear operator $A$ are more directly related to its Rayleigh Quotient. The Rayleigh Quotient informs us about the geometry of its vectors in high-dimensional spaces. Thus, we change our formulation and construct a doubly-weighted Laplace operator. This operator has the special property that it retains both edge and vertex information by explicitly weighing them during operation. For the coarse graph $\hat{G}$ with each vertex $\hat{v} \in \hat{V}$ weighted by <span style="color:#7F00FF;">$\gamma = |\pi^{-1}(\hat{v})|$</span>, let $\Gamma$ be the diagonal $n \times n$ vertex matrix with entries $\Gamma[r][r] = \gamma_{r}$, the doubly-weighted Laplace operator is then defined as,</p>

<p style="text-align:center;">$\mathcal{L} = $ <span style="color:steelblue;">$\Gamma^{-\frac{1}{2}}$</span>$\hat{L}$<span style="color:#7F00FF;">$\Gamma^{-\frac{1}{2}}$</span>$ = $<span style="color:steelblue;">$\underbrace{(P^{+}\Gamma^{-\frac{1}{2}})^{T}}_{\text{lift}}$</span>$L$<span style="color:#7F00FF;">$\underbrace{(P^{+}\Gamma^{-\frac{1}{2}})}_{\text{project}}$</span></p>

<p>This motivates us to use $\mathcal{L}$ as a similarity measure during the coarsening process.</p>


# Learning Graph Coarsening  

> <p><b>TLDR</b>- GOREN utilizes a GNN to assign edge weights by minimizing an unsupervised loss between functional mappings <span style="color:steelblue;">$\mathcal{F}(L,f)$</span> and <span style="color:#7F00FF;">$\mathcal{F}(\hat{L},\hat{f})$</span>. Weight adjustment yields improved quadratic and eigen error reduction.</p>

We now utilize the Rayleigh Quotient formulation to construct a framework for learning coarsening of graphs. The core of learning process is formed by the motivation to reset and learn edge weight assignments of the coarse graph.

### The GOREN Framework

Following previous section, we know that the usage of better Laplace operators leads to <span class="popup" onclick="myFunction('mynewPopup')">better-informed<span class="popuptext" id="mynewPopup"> strictly in the spectral sense.</span></span> weights. Using this insight, the setting develops a framework wherein the weight $\hat{w}(\hat{v},\hat{v}^{\prime})$ on the edge $(\hat{v},\hat{v}^{\prime})$ is predicted using a <em>weight-assignment function</em> $\mu(G_{A})$. Here, $G_{A}$ denotes the subgraph induced by a subset of vertices $A$. Note that there are two problems with $\mu(G_{A})$; (1) it is unclear as to what the stucture of this function should be and (2) the functional arguments can scale rapidly for large number of nodes in $A$. This is where <cite><a href="https://distill.pub/2021/gnn-intro/">Graph Neural Networks</a></cite> (GNNs) come in.

GNNs are employed as effective parameterizations to learn a collection of input graphs in unsupervised fashion. The weight-assignment map $\mu$ corresponds to a learnable neural network. This <span style="color:#7F00FF;">G</span>raph c<span style="color:#7F00FF;">O</span>arsening <span style="color:#7F00FF;">R</span>efinem<span style="color:#7F00FF;">E</span>nt <span style="color:#7F00FF;">N</span>etwork <span style="color:#7F00FF;">(GOREN)</span> reasons about local regions as supernodes and assigns a weight to each edge in the coarse graph. GOREN uses the <cite><a href="https://openreview.net/forum?id=ryGs6iA5Km">Graph Isomorphism Network</a></cite> (GIN) as its GNN model. All edge attributes of the coarse graph are initialized to 1. Weights assigned by the network are enforced to be positive using an additional ReLU output layer.

<p style="text-align:center;"><img src="/public/images/2022-03-25-coarsening/graph_gif.gif" style="width:640px;height:200px;display:inline-block;"></p>
<p style="text-align:center;color:silver;font-size:15px;margin-top:10px;">GOREN enables weight adjustment using similarity measures between $G$ and $\hat{G}$.</p>

GOREN is trained to minimize the distance between functional mappings <span style="color:steelblue;">$\mathcal{F}(L,f)$</span> and <span style="color:#7F00FF;">$\mathcal{F}(\hat{L},\hat{f})$</span> with $\hat{f}$ as the projection of $f$ and $k$ as the number of node atttributes,

<p style="text-align:center;">$
  \text{Loss}(L,\hat{L}) = \frac{1}{k}\sum_{i=1}^{k}|$<span style="color:steelblue;">$\underbrace{\mathcal{F}(L,f_{i})}_{\text{original graph}}$</span>$ - $<span style="color:#7F00FF;">$\underbrace{\mathcal{F}(\hat{L},\hat{f}_{i})}_{\text{coarsened graph}}$</span>$|$
</p>

<details>
<summary>
GOREN (8 lines)
</summary>
{% highlight js %}
def goren(graph, net):
    weights = net.forward(graph)
    coarse_graph = construct_graph(weights)
    l = compute_l(graph)
    l_hat = compute_l_hat(coarse_graph)
    loss = 0.5*(l - l_hat)**2
    loss = loss.mean().backward()
    return loss.data
{% endhighlight %}
</details>

Practical implementation of the algorithm uses Rayleigh formulation with $\mathcal{F}$ as the Rayleigh Quotient, $\hat{L}$ as the double-weighted Laplacian $\mathcal{L}$, and $\hat{f}$ as the lift-project mapping $\hat{f} = (P^{+})^{T}L$. At test time, a graph $G_{\text{test}}$ is passed through the network to predict edge weights for the new graph $\hat{G}_{\text{test}}$ by minimizing the loss iteratively.

### Weight Assignments & Adjustments

Primary component which rests behind the success of GOREN is its weight assignment and adjustment scheme. Using a GNN allows GOREN to reset and adjust edge weights. These weights capture local structure from neighborhoods of the original graph. This small trick can be applied on top of previous coarsening algorithms and improve node connectivity. 

<p style="text-align:center;"><img src="{{ site.url }}/public/images/2022-03-25-coarsening/ws.png" style="width:500px;height:320px;display:inline-block;" /><img src="{{ site.url }}/public/images/2022-03-25-coarsening/shape.png" style="width:500px;height:320px;display:inline-block;" /></p>

<p style="text-align:center;color:silver;font-size:15px;margin-top:10px;">GOREN outperforms MLP weight assignment in error reduction on (left) WS and (right) shape datasets.</p>

When compared to MLP-based weight assignment on various coarsening algorithms; namely random BaseLine (BL), <cite><a href="https://arxiv.org/pdf/1108.1310.pdf">Affinity</a></cite>, <cite><a href="https://epubs.siam.org/doi/abs/10.1137/090775087?journalCode=sjoce3">Algebraic Distance</a></cite>, <cite><a href="https://ieeexplore.ieee.org/document/4302760">Heavy edge matching</a></cite>, Local variation (in edges) and Local variation (in neighbors); GOREN demonstrates greater % improvement in error reduction. Especially on the shape dataset where MLPs fail to assign weights, GOREN presents significant minimization in error. This results in better connectivity of the coarsened graph with spectral properties preserved of the original graph.

# Closing Remarks  

> <p><b>TLDR</b>- Learning the coarsening process can be further improved by extending GOREN towards (1) learnable node assignment, (2) non-differentiable loss functions and (3) similarity with downstream tasks.</p>

We looked at graph coarsening in the presence of edge weights. A weight assignment and adjustment scheme is constructed using the GOREN framework. GNNs are trained in an unsupervised fashion to preserve the spectral properties of original graph. While our discussion has been restricted to the study of GOREN, we briefly extend the scope towards other future avenues.

### Beyond Graph Coarsening

An important aspect to consider is the node topology of graphs. While GOREN operates on node-coarsened structures, it would be interesting to extend the framework towards a full coarsening approach. Grouping in supernodes can be learned using a separate GNN following GOREN. Such algorithms would lead to the generation of end-to-end graph coarsening methods. This will reduce our dependence on handcrafted coarsening techniques.

Another line of work originates from different types of losses. GOREN utilizes differentiable loss functions including mappings. The study can be further extended to non-differentiable loss functions which require challenging computations, e.g- inverse Laplacian. These functions consist of mappings with hidden spectral features which are difficult to obtain.

Lastly, GOREN could explicitly reason about which properties are important to preserve. This would provide interesting insights into various metrics and their utility in downstream learning of coarsened graphs.

### Further Reading

1. <cite>Chen Cai, Dingkang Wang, Yusu Wang, <a href="https://openreview.net/pdf?id=uxpzitPEooJ">Graph Coarsening with Neural Networks</a>, ICLR 2021.</cite>

2. <cite>Andreas Loukas, Pierre Vandergheynst, <a href="https://arxiv.org/pdf/1802.07510.pdf">Spectrally approximating large graphs with smaller graphs</a>, ICLR 2018.</cite>

3. <cite>Gecia Bravo-Hermsdorf, Lee M. Gunderson, <a href="https://proceedings.neurips.cc/paper/2019/file/cd474f6341aeffd65f93084d0dae3453-Paper.pdf">A Unifying Framework for Spectrum-Preserving Graph Sparsification and Coarsening</a>, NeurIPS 2019.</cite>

4. <cite>Benjamin Sanchez-Lengeling, Emily Reif, Adam Pearce, Alexander B. Wiltschko, <a href="https://distill.pub/2021/gnn-intro/">A Gentle Introduction to Graph Neural Networks</a>, Distill 2021.</cite>

5. <cite>Ameya Daigavane, Balaraman Ravindran, Gaurav Aggarwal, <a href="https://distill.pub/2021/understanding-gnns/">Understanding Convolutions on Graphs</a>, Distill 2021.</cite>

6. <cite>Danijela Horak, Jürgen Jost, <a href="https://arxiv.org/abs/1105.2712">Spectra of combinatorial Laplace operators on simplicial complexes</a>, AIM 2013.</cite>

7. <cite>Andreas Loukas, <a href="https://arxiv.org/abs/1808.10650">Graph reduction with spectral and cut guarantees</a>, JMLR 2019.</cite>

8. <cite>Hermina Petric Maretic, Mireille EL Gheche, Giovanni Chierchia, Pascal Frossard, <a href="https://arxiv.org/abs/1906.02085">GOT: An Optimal Transport framework for Graph comparison</a>, NeurIPS 2019.</cite>

<style>
p {
  text-align: justify;
}

h1 {
  margin: 0;
  display: inline-block;
}

.center {
        margin: auto;
        width: 60%;
}

.popup {
  position: relative;
  display: inline;
  cursor: pointer;
  color: silver;
  border-bottom: 1px dashed silver;
  text-decoration: none;
}

.popup .popuptext {
  visibility: hidden;
  width: 160px;
  background-color: #555;
  color: #fff;
  text-align: center;
  border-radius: 6px;
  padding: 8px 0;
  position: absolute;
  z-index: 1;
  bottom: 125%;
  left: 50%;
  margin-left: -80px;
}

.popup .popuptext::after {
  content: "";
  position: absolute;
  top: 100%;
  left: 50%;
  margin-left: -5px;
  border-width: 5px;
  border-style: solid;
  border-color: #555 transparent transparent transparent;
}

.popup .show {
  visibility: visible;
  -webkit-animation: fadeIn 1s;
  animation: fadeIn 1s
}

@-webkit-keyframes fadeIn {
  from {opacity: 0;}
  to {opacity: 1;}
}

@keyframes fadeIn {
  from {opacity: 0;}
  to {opacity:1 ;}
}

.button {
  background-color: #7F00FF;
  border: none;
  border-radius: 20px;
  box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.19);
  color: white;
  display: inline-block;
  padding: 16px 32px;
  text-align: center;
  text-decoration: none;
  float: right;
  font-size: 16px;
}

.dropbtn {
  background-color: #4CAF50;
  color: white;
  padding: 16px;
  font-size: 16px;
  border: 1px solid #f1f1f1;
  border-radius: 20px;
  box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.19);
  cursor: pointer;
}

.dropdown {
  position: relative;
  display: inline-block;
  float:right;
}

.dropdown-content {
  display: none;
  position: inline-block;
  background-color: #f9f9f9;
  min-width: 320px;
  box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
  z-index: 1;
}

.dropdown-content a {
  color: black;
  padding: 12px 16px;
  text-decoration: none;
  display: block;
}

.dropdown-content a:hover {background-color: #f1f1f1}

.dropdown:hover .dropdown-content {
  display: block;
}

.dropdown:hover .dropbtn {
  background-color: #3e8e41;
}

.flip-card {
  background-color: transparent;
  display: inline-block;
  width: 200px;
  height: 200px;
  border: 1px solid #f1f1f1;
  border-radius: 20px;
  box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.19);
  perspective: 1000px;
}

.flip-card-inner {
  position: relative;
  width: 100%;
  height: 100%;
  text-align: center;
  transition: transform 0.8s;
  transform-style: preserve-3d;
}

.flip-card:hover .flip-card-inner {
  transform: rotateY(180deg);
}

.flip-card-front, .flip-card-back {
  position: absolute;
  width: 100%;
  height: 100%;
  -webkit-backface-visibility: hidden; /* Safari */
  backface-visibility: hidden;
}

.flip-card-front {
  border-radius: 20px;
  box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.19);
  background-color: #bbb;
  color: black;
}

.flip-card-back {
  border-radius: 20px;
  box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.19);
  background-color: dodgerblue;
  transform: rotateY(180deg);
}
</style>
<script>
function myFunction(id) {
  var popup = document.getElementById(id);
  popup.classList.toggle("show");
}
</script>

