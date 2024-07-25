# Graph Neural Networks: A suitable Alternative to MLPs in Latent 3D Medical Image Classification?


**Johannes Kiechle<sup>1,2,3,4</sup>**, Stefan M. Fischer<sup>1,2,3,4</sup>, Daniel M. Lang<sup>1,3</sup>, Lina Felsner<sup>1,3</sup>, Jan C. Peeken<sup>2,3</sup> and Julia A. Schnabel<sup>1,3,4,5</sup>

<sup>1</sup> Technical University of Munich, Germany \
<sup>2</sup> Klinikum rechts der Isar, Munich, Germany \
<sup>3</sup> Helmholtz Munich, Germany \
<sup>4</sup> Munich Center for Machine Learning \
<sup>5</sup> King's College London, United Kingdom


Accepted at [MICCAI 2024 - Workshop on GRaphs in biomedicAl Image anaLysis (GRAIL)](https://grail-miccai.github.io/) | [preprint](https://arxiv.org/pdf/2407.17219)

<p align="center">
  <img src="./figures/method.png" width="800"/>
</p>

**Abstract:** Recent studies have underscored the capabilities of natural imaging foundation models to serve as powerful feature extractors, even in a zero-shot setting for medical imaging data. Most commonly, a shallow multi-layer perceptron (MLP) is appended to the feature extractor to facilitate end-to-end learning and downstream prediction tasks such as classification, thus representing the *de facto* standard. However, as graph neural networks (GNNs) have become a practicable choice for various tasks in medical research in the recent past, we direct attention to the question of how effective GNNs are compared to MLP prediction heads for the task of 3D medical image classification, proposing them as a potential alternative. In our experiments, we devise a subject-level graph for each volumetric dataset instance. Therein latent representations of all slices in the volume, encoded through a DINOv2 pretrained vision transformer (ViT), constitute the nodes and their respective node features. We use public datasets to compare the classification heads numerically and evaluate various graph construction and graph convolution methods in our experiments. Our findings show enhancements of the GNN in classification performance and substantial improvements in runtime compared to an MLP prediction head. Additional robustness evaluations further validate the promising performance of the GNN, promoting them as a suitable alternative to traditional MLP classification heads.

**Keywords:** Classification · Graph Neural Networks · Graph Topology 