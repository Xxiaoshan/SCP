# SCP
Code implementation of “Real-Time Semi-Supervised Modulation Recognition Framework based on Signal Symbol Contrastive Prediction”

We have released our paper and our codebase to the community. In this article, we implement a semi-supervised modulation recognition algorithm based on symbol contrastive prediction. Existing modulation recognition algorithms are typically supervised and suffer severe performance degradation in the absence of labeled samples. For AMR tasks, a large amount of unlabeled data is often wasted, as it is easy to collect but not label radio signals. Therefore, we aim to leverage unlabeled samples to learn the modulation features of signals. Inspired by contrastive predictive coding, we adopt a pre-training and fine-tuning framework. 

The popular RML2016.10A and RML2018.01A datasets are used to evaluate the proposed method's advancement. We conduct empirical analyses comparing our approach to existing semi-supervised and supervised methods. The results demonstrate that SCP excels in semi-supervised modulation recognition, especially on the RML2018.01A dataset with longer signal lengths (1024).


