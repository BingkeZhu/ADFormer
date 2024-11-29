# ADFormer: Generalizable Few-Shot Anomaly Detection with Dual CNN-Transformer Architecture

## 💡 Introduction

<p align="center" style="border-radius: 10px">
  <img src="asset/adformer.png" width="90%" alt="adformer"/>
</p>

In Generalizable Few-Shot Anomaly Detection (GFSAD), a common model must be learned and shared across several categories, while simultaneously ensuring that the model is adaptable to new categories with a restricted number of normal images. While CNN-transformer architectures obtain high success in many vision tasks, the potential of CNN-transformer architectures in GFSAD is still to be discovered. In this paper, we introduce **ADFormer**, a dual CNN-transformer architecture that combines the strengths of CNNs and transformers, with the aim of learning discriminative features that have both local and global receptive fields. We also incorporate a self-supervised bipartite matching approach in ADFormer that reconstructs query images from support images, followed by detecting anomalies based on the high loss in reconstruction. Additionally, we present a consistency-enhanced loss to enhance the spatial and semantic consistency of features, thereby reducing the dependence on a large AD dataset for training. Experimental results show that ADFormer with consistency-enhanced loss significantly improves GFSAD performance. Compared to other anomaly detection methods, ADFormer outperforms considerably on the MVTec AD, MPDD, and VisA datasets.

## 🔥 Training 

```bash
python train.py
```

## 💻 Evaluation

```bash
python test.py
```

## 🤗Acknowledgements

- Thanks to [MaskFormer](https://github.com/facebookresearch/MaskFormer) for their wonderful work and code!

## 📖BibTeX

```
@article{zhu2024adformer,
  title={ADFormer: Generalizable Few-Shot Anomaly Detection with Dual CNN-Transformer Architecture},
  author={Zhu, Bingke and Gu, Zhaopeng and Zhu, Guibo and Chen, Yingying and Tang, Ming and Wang, Jinqiao},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2024},
  publisher={IEEE}
}
```

