# **Benchmarking and Boosting Transformers for Medical Image Classification**

This repository is based on and extends the original work **"Benchmarking and Boosting Transformers for Medical Image Classification"** by [DongAo Ma](https://github.com/Mda233) et al., presented at MICCAI 2022. The project benchmarks transformer-based models for medical image classification and explores pre-training and domain adaptation techniques to overcome the challenges of annotation scarcity in medical imaging.

You can find the original repository [here](https://github.com/Mda233/BenchmarkTransformers).

---

## **Overview**

This repository provides implementations, pre-trained models, and fine-tuning instructions for applying transformers to various medical image classification tasks. The project demonstrates:
1. The importance of pre-training for transformer models in medical imaging.
2. The benefits of self-supervised learning over supervised baselines.
3. Domain-adaptive pre-training as a method for bridging the gap between photographic and medical imaging.

---

## **Highlights**

1. **Transformers outperform CNNs with proper pre-training:**  
   Pre-trained transformers rival CNNs on medical imaging tasks but underperform when trained from scratch.
   
2. **Self-supervised learning (SSL) with masked image modeling (SimMIM):**  
   SimMIM-based self-supervised learning outperforms supervised pre-training baselines.

3. **Domain-adaptive pre-training with large-scale in-domain datasets:**  
   Models pre-trained on medical imaging datasets like X-rays significantly boost performance across tasks.

---

## **Major Contributions**

- Benchmarking supervised and self-supervised transformer models against CNNs.
- A practical pipeline for pre-training transformers using domain-adaptive SSL on large-scale medical datasets.
- Open-sourced pre-trained models and training scripts for reproducibility.

---

## **How to Use**

### **1. Pre-trained Models**
Download the pre-trained models used in this project from the links below:

| Category                | Backbone    | Dataset                  | Training Objective   | Model Link                                                                                                                                    |
|-------------------------|-------------|--------------------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Domain-adapted models** | Swin-Base   | ImageNet → X-rays (926K) | SimMIM → SimMIM      | [Download](https://zenodo.org/record/7101953/files/simmim_swinb_ImageNet_Xray926k.pth?download=1)                                             |
|                         | Swin-Base   | ImageNet → ChestX-ray14  | SimMIM → SimMIM      | [Download](https://zenodo.org/record/7101953/files/simmim_swinb_ImageNet_ChestXray14.pth?download=1)                                         |
| **In-domain models**    | Swin-Base   | X-rays (926K)            | SimMIM               | [Download](https://zenodo.org/record/7101953/files/simmim_swinb_Scratch_Xray926k.pth?download=1)                                             |
|                         | Swin-Base   | ChestX-ray14             | SimMIM               | [Download](https://zenodo.org/record/7101953/files/simmim_swinb_Scratch_ChestXray14.pth?download=1)                                          |

### **2. Fine-Tuning**

Use the provided script for fine-tuning a pre-trained model on your target dataset. Example:

```bash
python main_classification.py --data_set ChestXray14 \
  --model swin_base \
  --init simmim \
  --pretrained_weights [PATH_TO_MODEL]/simmim_swinb_ImageNet_Xray926k.pth \
  --data_dir [PATH_TO_DATASET] \
  --train_list dataset/Xray14_train_official.txt \
  --val_list dataset/Xray14_val_official.txt \
  --test_list dataset/Xray14_test_official.txt \
  --lr 0.01 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64


---

## **Acknowledgments**

This project builds upon the original work **"Benchmarking and Boosting Transformers for Medical Image Classification"** by [DongAo Ma](https://github.com/Mda233) et al. The original research was conducted at Arizona State University and Mayo Clinic, and presented at the MICCAI 2022 DART Workshop.

The original research was supported in part by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and in part by the NIH under Award Number R01HL128785. This work also utilized GPUs provided by ASU Research Computing and the Extreme Science and Engineering Discovery Environment (XSEDE) funded by the National Science Foundation (NSF) under grant numbers ACI-1548562, ACI-1928147, and ACI-2005632.

For further details, refer to the original repository:
- [Original GitHub Repository](https://github.com/Mda233/BenchmarkTransformers)
- [Original Paper](https://link.springer.com/chapter/10.1007/978-3-031-16852-9_2)

If you use this repository or pre-trained weights for your research, please acknowledge the original authors.

---

## **Citation**

Please cite the original work if you use this code or pre-trained models in your research:

```bibtex
@inproceedings{Ma2022Benchmarking,
    title="Benchmarking and Boosting Transformers for Medical Image Classification",
    author="Ma, DongAo and Hosseinzadeh Taher, Mohammad Reza and Pang, Jiaxuan and Islam, Nahid UI and Haghighi, Fatemeh and Gotway, Michael B and Liang, Jianming",
    booktitle="Domain Adaptation and Representation Transfer",
    year="2022",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="12--22",
    isbn="978-3-031-16852-9"
}
