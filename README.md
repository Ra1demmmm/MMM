
# MMM: A Unified Weakly-supervised Anomaly Detection Framework for Multi-distributional Data

This repo contains source codes for the paper [**MMM: A Unified Weakly-supervised Anomaly Detection Framework for Multi-distributional Data**](https://) submitted to IEEE Transactions on Knowledge and Data Engineering.

## Abstract

Weakly-Supervised Anomaly Detection (WSAD) has garnered increasing research interest in recent years, as it enables superior detection performance while demanding only a small fraction of labeled data. 
However, existing WSAD methods face two major limitations. From the data aspect, they struggle to detect anomalies between normal clusters or collective anomalies due to overlooking the multi-distribution and complex manifolds of real-world data.
From the label aspect, they fall short of detecting unknown anomalies because of the label-insufficiency and anomaly contamination. To address these issues, we propose MMM, a unified WSAD framework for multi-distributional data. The framework consists of three components: a Multi-distribution data modeler captures latent representations of complex data distributions, followed by a Multiform feature extractor that extracts multiple underlying features from the modeler, highlighting the characteristics of potential anomalies. Finally, a Multi-strategy anomaly score estimator converts these features into anomaly scores, with the aid of a novel training approach with three strategies that maximize the utility of both data and labels. Experimental results show that MMM achieves superior performance and robustness compared to state-of-the-art WSAD methods, while providing interpretable results that facilitate practical anomaly analysis.

## Software Requirement

* Python 3.10
* numpy 1.26.4
* pandas 2.3.0
* tqdm 4.66.2
* scikit-learn 1.4.1
* scipy 1.12.0
* torch 1.13.0



