---
layout: page
title: ICLR 2022 Blog Track (OLD)
tags: [proposal, call]
authors: Bubeck, Sebastien (Microsoft); Dobre, David (Mila); Gauthier, Charlie (Mila); Gidel, Gauthier (Mila); Vernade, Claire (DeepMind)
---

<br>
<p align="center">
  <img src="{{ site.url }}/public/images/2021-09-01-sample-submission/ICLR-logo.png" width="50%" alt="ICLR Logo">
</p>

## Important Information

- For all editions of the ICLR Blog Track after 2022, please visit this website: [https://iclr-blogposts.github.io/2024/about](https://iclr-blogposts.github.io/2023/about)
- The track has concluded and accepted blogposts are viewable [here]({{ site.url }}/blog)!
- We've released a video talking about this track in more detail

We would like to thank everyone who took part in this experiment and for making it a success!

<p align="center"><iframe width="560" height="315" src="https://www.youtube.com/embed/pDgvYpRfiJw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></p>

## Contents

- [Accepted Posts](#accepted-posts)
- [Motivation](#motivation)
- [Submissions](#submissions)
- [Organizers](#organizers)

## Accepted Posts

**[An Understanding of Learning from Demonstrations for Neural Text Generation]({% post_url 2022-03-25-text-gen-via-lfd %})**
: _Kantharaju, Pavan, Smart Information Flow Technologies; Sankar, Aiswarya, Independent_

**[Auction Learning as a Two Player Game: GANs (?) for Mechanism Design]({% post_url 2022-03-25-two-player-auction-learning %})**
: _Curry, Michael J., University of Maryland; Reusche, Daniel_

**[Deep Neural Nets: 33 years ago and 33 years from now (Invited Post)]({% post_url 2022-03-26-lecun1989 %})**
: _Karpathy, Andrej_

**[A Deeper Look at Zero-Cost Proxies for Lightweight NAS]({% post_url 2022-03-25-zero-cost-proxies %})**
: _White, Colin; Khodak, Mikhail; Tu, Renbo; Shah, Shital; Bubeck, Sébastien; Dey, Debadeepta_

**[Discovering Non-Monotonic Autoregressive Ordering for Text Generation Models using Sinkhorn Distributions]({% post_url 2022-03-25-non-monotonic-autoregressive-ordering %})**
: _Kumar, Ashutosh_

**[Does Adam Converge and When?]({% post_url 2022-03-25-does-adam %})**
: _Zhang, Yushun; Chen, Congliang; Luo, Zhi-Quan_

**[Euclidean geometry meets graph, a geometric deep learning perspective]({% post_url 2022-03-25-euclidean_geometric_graph %})**
: _Wang, Zichen, Amazon Web Services; Shi, Yunzhi, Amazon Web Services; Chen, Xin, Amazon Web Services_

**[Generating Molecular Conformations via Normalizing Flows and Neural ODEs]({% post_url 2022-03-25-conformation-generation %})**
: _Mukundh Murthy, Nikhil Devraj_

**[Knowledge Graph Papers @ ICLR 2021]({% post_url 2022-03-25-kgs %})**
: _Galkin, Mikhail (Mila & McGill University)_

**[Learning to Coarsen Graphs with Graph Neural Networks]({% post_url 2022-03-25-coarsening %})**
: _Suri, Karush_

**[Looking at the Performer from a Hopfield Point of View]({% post_url 2022-03-25-Looking-at-the-Performer-from-a-Hopfield-point-of-view %})**
: _Brandstetter J. and Ramsauer H. and Holzleitner M. and Hochreiter S. and Schäfl B._

**[Normalization is dead, long live normalization!]({% post_url 2022-03-25-unnormalized-resnets %})**
: _Hoedt, Pieter-Jan; Hochreiter, Sepp; Klambauer, Günter_

**[On Dyadic Fairness: Exploring and Mitigating Bias in Graph Connections]({% post_url 2022-03-25-dyadic-fairness %})**
: _Subramonian, Arjun_

**[PPLM Revisited: Steering and Beaming a Lumbering Mammoth to Control Text Generation]({% post_url 2022-03-25-PPLM %})**
: _Nguyen, Van Bach; Trienes, Jan; Nauta, Meike; Pathak, Shreyasi; Youssef, Paul; Imangaliyev, Sultan; Schlötterer, Jörg; Seifert, Christin_

**[Recent Advances in Deep Learning for Routing Problems]({% post_url 2022-03-25-deep-learning-for-routing-problems %})**
: _Joshi, Chaitanya K.; Anand, Rishabh_

**[Representation Change in Model-Agnostic Meta-Learning]({% post_url 2022-03-25-representation-change-in-model-agnostic-meta-learning %})**
: _Goerttler, Thomas (TU Berlin); Müller, Luis (TU Berlin); Obermayer, Klaus (TU Berlin)_

**[Rethinking ValueDice - Does It Really Improve Performance?]({% post_url 2022-03-25-rethinking-valuedice %})**
: _Ziniu, Li, CUHKSZ; Tian, Xu, NJU; Yang, Yu, NJU; Zhi-Quan, Luo, CUHKSZ_

**[Symbolic Binding in Neural Networks through Factorized Memory Systems]({% post_url 2022-03-25-emergent-symbols %})**
: _Ameya Daigavane, Ansh Khurana, Shweta Bhardwaj, Gaurav Aggarwal_

**[The 37 Implementation Details of Proximal Policy Optimization]({% post_url 2022-03-25-ppo-implementation-details %})**
: _Huang, Shengyi; Dossa, Rousslan Fernand Julien; Raffin, Antonin; Kanervisto, Anssi; Wang, Weixun_

**[The Annotated S4]({% post_url 2022-03-25-annotated-s4 %})**
: _Rush, Alexander; Karamcheti, Sidd_

**[Understanding Few-Shot Multi-Task Representation Learning Theory]({% post_url 2022-03-25-understanding_mtr_meta %})**
: _Bouniot, Quentin; Redko, Ievgen_


## Motivation

The Machine Learning community is currently experiencing a
[reproducibility crisis](https://neuripsconf.medium.com/designing-the-reproducibility-program-for-neurips-2020-7fcccaa5c6ad)
and a reviewing crisis [[Littman, 2021]](#Litt). Because of the highly competitive and noisy
reviewing process of ML conferences [[Tran et al., 2020]](#Tran), researchers have an incentive to
oversell their results, slowing down the progress and diminishing the
integrity of the scientific community. Moreover with the growing number
of papers published and submitted at the main ML conferences [[Lin et al., 2020]](#Lin), it has
become more challenging to keep track of the latest advances in the
field.

Blog posts are becoming an increasingly popular and useful way to talk
about science [[Brown and Woolston, 2018]](#Brow).
They offer substantial value to the scientific community
by providing a flexible platform to foster open, human, and transparent
discussions about new insights or limitations of a scientific
publication. However, because they are not as recognized as standard
scientific publications, only a minority of researchers manage to
maintain an active blog and get visibility for their efforts. Many are
well-established researchers ([Francis Bach](https://francisbach.com/),
[Ben Recht](https://www.argmin.net/), [Ferenc
Huszár](https://www.inference.vc/), [Lilian
Weng](https://lilianweng.github.io/lil-log/)) or big corporations that
leverage entire teams of graphic designers designer and writers to
polish their blogs ([Facebook AI](https://ai.facebook.com/blog/?page=1),
[Google AI](https://ai.googleblog.com/),
[DeepMind](https://deepmind.com/blog),
[OpenAI](https://openai.com/blog/)). As a result, the incentives for
writing scientific blog posts are largely personal; it is unreasonable
to expect a significant portion of the machine learning community to
contribute to such an initiative when everyone is trying to establish
themselves through publications.

> You can read more on our [about]({% link about.md %}) page.

### A Blog Post Conference Track

Our goal is to create a formal call for blog posts at ICLR to
incentivize and reward researchers to review past work and summarize the
outcomes, develop new intuitions, or highlight some shortcomings. A very
influential initiative of this kind happened after the second world war
in France. Because of the lack of up-to-date textbooks, a collective of
mathematicians under the pseudonym Nicolas Bourbaki [[Halmos 1957]](#Halm), decided to start a
series of textbooks about the foundations of mathematics [[Bourbaki, 1939]](#Bour).
In the same vein, we aim at providing a new way to summarize scientific knowledge in
the ML community.

Due to the large diversity of topics that can be discussed in a blog
post, we decided to restrict the range of topics for this call for blog
posts. We identified that the blog posts that would bring to most value
to the community and the conference would be posts that distill and
discuss *previously published papers*.

### A call for blog posts discussing work previously published at ICLR

The format and process for this blog post track is as follows:

-   Write a post about a paper previously published at ICLR, with the
    constraint that one cannot write a blog post on work that they have
    a conflict of interest with. This implies that one cannot review
    their own work, or work originating from their institution or
    company. We want to foster productive discussion about *ideas*, and
    prevent posts that intentionally aim to help or hurt individuals or
    institutions.

-   Blogs will be peer-reviewed (double-blind, see
    Section <a href="#sub:sub_process" data-reference-type="ref" data-reference="sub:sub_process">2.5</a>)
    for quality and novelty of the content: clarity and pedagogy of the
    exposition, new theoretical or practical insights,
    reproduction/extension of experiments, etc.

-   The posts will be published under a unified template (see
    Section <a href="#sub:sub_format" data-reference-type="ref" data-reference="sub:sub_format">2.4</a>
    and
    Section <a href="#sub:sub_process" data-reference-type="ref" data-reference="sub:sub_process">2.5</a>)
    and hosted on the conference website or our own Github page.


## Submissions

**Note: The track has concluded and we are not accepting any more submissions!**

Our goal is to avoid heavily engineered, professionally-made
blog-posts---Such as the “100+ hours” mentioned as a standard by the [Distill
  guidelines](https://distill.pub/journal/)---to entice ideas and clear writing rather than dynamic
visualizations or embedded javascript engines.

As a result, we restrict submissions to the Markdown format. We believe
this is a good trade-off between complexity and flexibility. Markdown
enables users to easily embed media such as images, gifs, audio, and
video as well as write mathematical equations using MathJax, without
requiring users to know how to create HTML web pages. This (mostly)
static format is also fairly portable; users can download the blog post
without much effort for offline reading or archival purposes. More
importantly, this format can be easily hosted and maintained through
GitHub.

> Please checkout the <a href="submitting">submitting</a> section for a detailed overview on the
> process of creating and submitting a blog post.


## Organizers

&nbsp;

<ul class="image-list-small">
  <li>
    <a style="background-image: url({{site.url}}/public/images/organizers/gg.jpg);"></a>
    <div class="details">
      <h3>Gauthier Gidel</h3>
      <p class="image-author">gidelgau [ at ] mila.quebec</p>
    </div>
  </li>
  <li>
    <a style="background-image: url({{site.url}}/public/images/organizers/cg.jpg);"></a>
    <div class="details">
      <h3>Charlier Gauthier</h3>
      <p class="image-author">charlie.gauthier [ at ] umontreal.ca</p>
    </div>
  </li>
  <li>
    <a style="background-image: url({{site.url}}/public/images/organizers/dd.jpg);"></a>
    <div class="details">
      <h3>David Dobre</h3>
      <p class="image-author">david-a.dobre [ at ] mila.quebec</p>
    </div>
  </li>
  <li>
    <a style="background-image: url('{{site.url}}/public/images/organizers/sb.jpg');"></a>
    <div class="details">
      <h3>Sébastien Bubeck</h3>
      <p class="image-author">sebubeck [ at ] microsoft.com</p>
    </div>
  </li>
  <li>
    <a style="background-image: url('{{site.url}}/public/images/organizers/cv.jpg');"></a>
    <div class="details">
      <h3>Claire Vernade</h3>
      <p class="image-author">vernade [ at ] deepmind.com</p>
    </div>
  </li>
</ul>

---

## References

<a name="Litt">Michael L Littman. Collusion rings threaten the integrity of computer science research. Communications of the ACM, 2021.</a>

<a name="Tran">David Tran, Alex Valtchanov, Keshav Ganapathy, Raymond Feng, Eric Slud, Micah Goldblum, and Tom Goldstein. An open review of openreview: A critical analysis of the machine learning conference review process. arXiv, 2020. </a>

<a name="Lin">Hsuan-Tien Lin, Maria-Florina Balcan, Raia Hadsell, and Marc’Aurelio Ranzato. What we learned from neurips2020 reviewing process. Medium https://medium.com/@NeurIPSConf/what-we-learned-from-neurips-2020-reviewing-process-e24549eea38f, 2020. </a>

<a name="Brow">Eryn Brown and Chris Woolston. Why science blogging still matters. Nature, 2018.</a>

<a name="Halm">Paul R Halmos. Nicolas bourbaki. Scientific American, 1957.<a>

<a name="Bour">Nicolas Bourbaki. Elements of mathematics. Éditions Hermann, 1939.</a>





