Here is the full text of the paper, formatted for readability with corrected mathematical equations and structured tables based on the provided document.

# Generalized zero-shot domain adaptation with target unseen class prototype learning

**Xiao Li • Min Fang • Bo Chen**  
*Neural Computing and Applications (2022) 34:17793–17807*  
Received: 15 July 2021 / Accepted: 9 May 2022 / Published online: 4 June 2022

---

## Abstract
Generalized zero-shot domain adaptation (GZSDA) aims to classify samples from seen and unseen classes in a target domain by utilizing labeled data for all classes from a source domain and labeled data from seen classes in the target domain. GZSDA is more challenging than zero-shot learning or domain adaptation problems. We aim to learn prototypes for unseen classes in the target domain. The test samples can be classified into one of the seen and unseen classes based on the distance with the prototypes for seen and unseen classes in the target domain. Therefore, we propose a generalized zero-shot domain adaptation with a target unseen class prototype learning method (TUPL). We project the source samples and the target samples into a common subspace by making the samples of the same class near to cope with the domain difference. To strengthen the intra-class compactness of the samples, we pull samples closer to their class prototypes while maintaining data variance, learning discriminative representations in the subspace. Then, we learn the target unseen class prototypes by the relationships of the source and target domains and the relationships of the seen and unseen classes to get more accurate ones. The evaluations on the GZSDA datasets show that TUPL outperforms existing methods.

**Keywords:** Generalized zero-shot domain adaptation • Target unseen class prototypes • Domain alignment • Discrimination term

---

## 1 Introduction
Domain adaptation (DA) problem poses a major obstacle for transferring knowledge from a labeled source domain to an unlabeled target domain. The reason is that the shift in data distributions across different domains exists, which hinders the generalization of predictive models to target tasks. For example, we have many labeled gray images from the source domain. We aim to classify the color images from the target domain. Although the images from the source domain and the target domain are from the same classes, they follow different distributions. The classifier trained on the source domain samples cannot be directly applied to the target domain classification tasks. Many DA methods have been proposed to address this issue. In DA, the source and the target domains share the same label space.

In this study, we mainly focus on the more challenging setting of DA, the generalized zero-shot domain adaptation problem (GZSDA), where the label spaces of them are not the same. For the target domain, there are some labeled samples from seen classes. The task is to classify the unlabeled samples from seen and unseen classes which is the generalized zero-shot classification (GZSC). However, there are no semantic representations of the seen and unseen classes. Fortunately, the samples of all classes (seen classes and unseen classes) exist in the source domain. Thus, we can utilize labeled samples in the source domain and labeled samples from seen classes in the target domain to classify samples from seen and unseen classes in the target domain. The domain shift problem needs to be addressed. Thus, the GZSDA task to be solved in this paper is the combination of the GZSC and DA tasks.

GZSDA task is widespread in reality. For example, we aim to perform the cartoon image classification, including seen classes and new classes, in the target domain. However, there are no cartoon images of the new classes in the existing labeled target data. Fortunately, there is a source domain dataset where the images are collected from the Internet which contains the new classes. DA method cannot be applied to the problem because the source domain contains new classes and seen classes but the target domain only contains seen classes, the objective is to classify samples from seen and new classes in the target domain. Neither DA nor ZSC algorithms can be directly applied to this problem.

ZSC methods aim to classify the unseen samples for which no training samples are available during training. The labeled seen samples and the seen and unseen class representations are utilized to train the ZSL models. There are no intersections between seen and unseen classes. In GZSC setting, the test set contains seen and unseen samples instead of only seen samples. Both ZSL and GZSL require the class-level representations of seen and unseen classes. In GZSDA, there are no these information.

Recently, some zero-shot domain adaptation (ZSDA) methods have been proposed. ZSDA refers to that many labeled samples from seen and unseen classes are in the source domain, while only some labeled samples from seen classes are in the target domain. The target data to be classified are from unseen classes. ZSDA aims to classify these samples. The methods proposed require paired task-irrelevant data from the source and target domains but such paired data may not exist in most cases. GZSDA is more practical and difficult than ZSDA which aims to perform the recognition in the seen and unseen class spaces. The key challenge is the lack of unseen samples in the target domain. We aim to cope with the GZSDA problem by learning the target unseen class prototypes with the help of the labeled data for all classes from the source domain and labeled data from seen classes in the target domain.

We propose a method, generalized zero-shot domain adaptation with target unseen class prototype learning (TUPL). With the learned target unseen class prototypes, the test samples can be classified into one of the unseen classes based on the distances to them. Thus, the purpose of the paper is to learn them. The source seen (unseen) class prototypes can be calculated by the mean value of the seen samples of the same seen (unseen) class. The target seen class prototypes can be obtained in the same way. They cannot be directly used to learn the target unseen class prototypes because the domain difference exists across the two domains. Therefore, the first thing that we need to do is to learn a common subspace of the two domains.

We project the source samples and target samples to the common subspace following the same spirits of previous domain adaptation approaches. The projected samples from the same class in the subspace are pulled closer to minimize the domain difference. To learn the discriminative representations in the subspace, the projected samples are made near to the class prototypes of their classes and the data variance is maintained. Thus, in the subspace, we learn the target unseen class prototypes by the relationships of the source and target domains and the relationships of the seen and unseen classes. We train two networks simultaneously to capture the two relationships. Based on them, we can get reliable target unseen class prototypes. We classify the test samples based on their distances to the target seen and unseen class prototypes.

We can see from the experiments that TUPL achieves promising results. The contributions are as follows:
1. The generalized zero-shot domain adaptation problem, a more challenging setting of domain adaptation problem, is proposed. For dealing with the realistic GZSDA tasks, we propose to learn the target unseen class prototypes with the help of source seen and unseen data and target seen data.
2. We learn the common subspace to alleviate the domain shift problems. The domain alignment is performed by making the projected samples of the same class near. Meanwhile, we add a class center regularization term by making the samples near to their class prototypes to preserve the intra-class compactness.
3. Learning the target unseen class prototypes in the subspace is the main purpose. They are obtained not only by the relationships of source and target domains but also by the relationships of seen and unseen classes. Through the process, we can learn better target unseen class prototypes.
4. TUPL is evaluated on the GZSDA datasets. The results show that TUPL can overcome the domain shift problem, learn accurate target unseen class prototypes, and improve obviously than existing methods for GZSDA tasks.

The paper is organized as follows. The related works are discussed in Sect. 2. Section 3 presents a novel GZSDA method with target unseen class prototype learning (TUPL). Section 4 reports the experimental results. Section 5 discusses the conclusions.

---

## 2 Related works
GZSDA is more difficult than domain adaptation problems since the target unseen samples are missing. GZSDA aims to classify the test samples which are from seen or unseen classes in the target domain. We briefly review the related DA and GZSL methods.

**Domain adaptation:** These methods aim to apply the source domain knowledge to the target domain. The main point is to narrow down the domain difference between the source and target domains. A number of domain adaptation methods aim to project the source and target samples to a common subspace by minimizing a divergence metric, e.g., maximum mean discrepancy (MMD) metric and covariance to reduce the differences. Some researchers pay close attention to learning adaptive deep neural networks with adaptation layers to match the distributions between the source and target domains. A number of methods incorporate adversarial learning into the domain adaptation framework. These methods aim to learn domain-invariant features by training a feature learning network and a domain discriminator. However, these DA methods cannot be directly applied to the GZSDA, because they suppose the label spaces of the two domains are the same, which is not in line with the real-world applications. The proposed method aims to deal with a more difficult GZSDA problem, where the target unseen samples are missing.

**Zero-shot learning:** These methods rely on the shared information of the seen and unseen classes. The main point is to utilize the knowledge of the seen classes to recognize the unseen samples. Many ZSL methods have been proposed: semantic space methods, visual space methods, and common subspace methods. ZSL assumes the test samples are all from the unseen classes which is not in line with the real-world applications. The test samples may be from seen or unseen classes.

**Generalized zero-shot learning:** These methods aim to classify the test samples which contain seen and unseen samples. Direct application of ZSL methods in the GZSL problem leads to bias to the seen classes problem. Some methods aim to train generative models to generate unseen samples, such as generative adversarial networks (GAN) or variational autoencoders (VAE). In GZSDA, we do not have the semantic representations of the seen and unseen classes. These GZSL methods cannot be directly applied to GZSDA tasks.

In the context of prototype-based learning methods, learning vector quantization method approximates continuous functions of vectorial variables by a fixed number of codebook vectors. Snell et al. proposed prototypical networks to learn the prototypes of each class for few-shot classification. We focus on learning the target unseen prototypes for the GZSDA task. In GZSDA, we have a large amount of source seen and unseen samples. We utilize the source seen and unseen samples and the target seen samples to learn the target unseen class prototypes.

---

## 3 Proposed approach
We define the source domain data as $D_s = \{X_{sc}, Y_{sc}, X_{su}, Y_{su}\}$. $X_{sc} \in \mathbb{R}^{n_{sc} \times d}$ is the seen data matrix and $X_{su} \in \mathbb{R}^{n_{su} \times d}$ is the unseen data matrix with $d$ as the dimension and $n_{sc}$ as the number of seen samples and $n_{su}$ unseen samples. $Y_{sc} \in \mathbb{R}^{n_{sc} \times 1}$ and $Y_{su} \in \mathbb{R}^{n_{su} \times 1}$ are the seen and unseen labels from $\{1, \dots, C, C+1, \dots, C+U\}$. A target domain $D_t = \{X_{tc}, Y_{tc}\}$ is defined. $X_{tc} \in \mathbb{R}^{n_{tc} \times d}$ and $Y_{tc} \in \mathbb{R}^{n_{tc} \times 1}$ are the seen data and the seen labels which are from the seen classes $\{1, \dots, C\}$. $n_{tc}$ is the number of seen samples. $X_{te}$ are the target unlabeled seen and unseen samples.

The source samples and the target samples are mapped into the shared subspace. The samples are pulled closer to the samples of the same kind and their class prototypes to narrow down the domain shift and increase the compactness of the samples of the same class. When the target unseen class prototypes are obtained, we can classify the test samples based on the distance to the target seen and unseen class prototypes. Therefore, the emphasis of the method is learning the target unseen class prototypes.

### 3.1 Learning the common subspace
We learn the common subspace of the source and target domains by performing the domain alignment and preserving the intra-class compactness, simultaneously. The objective is:
$$L = L_{dt} + \alpha L_{di} \quad (1)$$
where $L_{dt}$ is the domain alignment term; $L_{di}$ is the class center regularization term; $\alpha$ is the parameter.

#### 3.1.1 Domain alignment
The source seen and unseen samples along with the target seen samples are mapped into a common subspace with the mapping function $f(\cdot)$. For simplicity, we assume $f(\cdot)$ as a linear function with $f(X) = XW$.
We expect the samples of the same kind are closer no matter which domains they are from. The similarity $Q_{i,j}$ of any two samples is defined as:
$$Q_{i,j} = \begin{cases} e^{-\|X_i - X_j\|^2} & X_i \text{ and } X_j \text{ are from the same class.} \\ 0 & \text{otherwise.} \end{cases} \quad (2)$$
The samples from the same class are pulled closer by minimizing Objective (3):
$$L_{da} = \sum_{i=1}^n Q_{i,j} \|X_i W - X_j W\|_2^2 = tr(W^T X^T L X W) \quad (3)$$
in which $L$ is: $L = D - Q$ (4)
where $X$ is the feature matrix of the source seen and unseen data, and the target seen data, $X = [X_{sc}; X_{su}; X_{tc}]$. $W \in \mathbb{R}^{d \times p}$. $p$ is the dimensionality of the common subspace. $D$ is a diagonal matrix, $D = \text{diag}(d_1, \dots, d_n)$, where $d_i$ is the sum of the $i$-th row of $Q$, $n = n_{sc} + n_{su} + n_{tc}$.

#### 3.1.2 Class center regularization term
To increase the intra-compactness of the samples, we make the samples near their class centers (prototypes).
The source seen class prototype of the $i$-th seen class is:
$$P_i^s = \frac{1}{n_i} \sum_{j=1}^{n_i} X_j^* \quad (5)$$
We reduce the distance between the samples and their class prototypes in the subspace:
$$L_{ds} = \sum_{i=1}^{n_{sc}} \|X_i^{sc} W - P_j^s W\|_2^2 + \sum_{i=1}^{n_{su}} \|X_i^{su} W - P_j^u W\|_2^2 \quad (6)$$
Writing Objective (6) in matrix form:
$$L_{ds} = \|X_s W - A_{ohs} W\|_F^2 \quad (7)$$
where $A_{ohs} = Y_{ohs} P$, $Y = [Y_{sc}; Y_{su}]$, $Y_{ohs}$ is the one-hot vector of $Y$, $P = [P_{sc}; P_{su}]$.
Similarly, we reduce the distances of the target seen samples and their target seen class prototypes:
$$L_{dt} = \sum_{i=1}^{n_{tc}} \|X_i^{tc} W - R_j^s W\|_2^2 = \|X_{tc} W - A_{oht} W\|_F^2 \quad (8, 9)$$
Combining them, the class center regularization term is:
$$L_{di} = \|X W - A_{oh} W\|_F^2 \quad (10)$$
The whole loss function can be formulated as:
$$\arg\min_W tr(W^T X^T L X W) + \alpha \|X W - A_{oh} W\|_F^2 \quad (11)$$

#### 3.1.3 Optimization
The optimization process can be written as:
$$\arg\min_W tr(W^T X^T L X W) + \alpha tr(W^T M W) \quad \text{s.t. } W^T X^T H X W = I \quad (12)$$
where $M = (X - A_{oh})^T (X - A_{oh})$. We add the constraints $W^T X^T H X W = I$ to avoid the trivial solution ($W=0$). $H = I - \frac{1}{n}\mathbf{1}$ is the centering matrix. We maximally preserve the data variance with the constraints.
Let $Q$ be the Lagrange multiplier, we take the Lagrangian of the function:
$$Z = tr(W^T X^T L X W) + \alpha tr(W^T M W) - tr((W^T X^T H X W - I)Q) \quad (13)$$
Setting the derivative of $Z$ w.r.t. $W$ to zero:
$$X^T L X W + \alpha M W - X^T H X W Q = 0 \quad (15)$$
$W$ can be obtained by solving the eigendecomposition of $(X^T L X + \alpha M)^{-1} X^T H X$ (18) for the $p$ biggest eigenvectors.
With the learned projection function, we can get the prototypes in the subspace:
$$P_{ws} = P_s W, \quad P_{wu} = P_u W, \quad R_{ws} = R_s W \quad (19)$$

### 3.2 Learning the target unseen class prototypes
Since the target unseen samples are unobtainable, the target unseen class prototypes $R_{wu}$ are unable to get directly. We learn $R_{wu}$ by the source–target relationships $\phi(\cdot)$ and the seen–unseen relationships $\psi(\cdot)$, simultaneously.

$\phi(\cdot)$ is the mapping function from source to target domain. We train an MLP:
$$L_{mlp1} = \|\phi(P_{ws}) - R_{ws}\|_F^2 \quad (20)$$
$$\phi(P_{ws}) = f_1(f_1(P_{ws} W_1) W_2) \quad (21)$$
We get the target unseen class prototypes with $\phi$: $\hat{R}_{wu} = \phi(P_{wu})$ (22)

In addition, we get the seen–unseen relationships $\psi(\cdot)$ by training another MLP:
$$L_{mlp2} = \|\psi(P_{ws}) - P_{wu}\|_F^2 \quad (23)$$
$$\psi(P_{ws}) = f_2(W_4 f_2(W_3 P_{ws})) \quad (24)$$
We obtain the target unseen class prototypes with $\psi$: $\tilde{R}_{wu} = \psi(R_{ws})$ (25)

We restrict the generated unseen class prototypes by the two methods to be the same:
$$L_s = \|\psi(R_{ws}) - \phi(P_{wu})\|_F^2 \quad (26)$$
The overall objective function is:
$$\arg\min_{\psi, \phi} \|\phi(P_{ws}) - R_{ws}\|_F^2 + \|\psi(P_{ws}) - P_{wu}\|_F^2 + \|\psi(R_{ws}) - \phi(P_{wu})\|_F^2 \quad (27)$$

Finally, the unlabeled target samples $X_{te}$ are projected to the subspace: $X_{te}^w = X_{te} W$ (28). We classify them by calculating their distances to the target seen and unseen class prototypes $R = [R_{ws}; R_{wu}]$:
$$y = \arg\min_j \text{Dist}(x_{te}^w, R_j) \quad (29)$$

---

## 4 Experiments

### 4.1 Experimental settings
**Datasets:**
*   **Office–Home dataset:** 4 domains (Artistic, Clipart, Product, Real-World) with 65 classes (35 seen, 30 unseen).
*   **Office31 dataset:** 3 domains (Amazon, Webcam, DSLR) with 31 classes (16 seen, 15 unseen).
*   **ImageCLEFDA dataset:** 3 domains (Caltech, ImageNet, Pascal) with 12 shared classes (6 seen, 6 unseen).

**Evaluation Metric:** Harmonic mean $H = \frac{2 \times Acc_s \times Acc_u}{Acc_s + Acc_u}$ (30).

**Compared Methods:** 1-NN, UUDAZS, LDA, MECA, ETN, BiDiLEL, LisGAN. ResNet50 is used for feature extraction.

### 4.2 Experimental results on Office–Home dataset
TUPL achieves much better performance on the Office-Home dataset than all compared methods. TUPL not only considers the domain difference problem by performing domain alignment but also learns discriminative representations by making them near to the class prototypes. Learning accurate unseen class prototypes for the target domain is significant for GZSDA tasks.

**Table 2: Office-Home (Source: A)**
| Method | A→C (Acc_s / Acc_u / H) | A→P (Acc_s / Acc_u / H) | A→R (Acc_s / Acc_u / H) |
| :--- | :--- | :--- | :--- |
| 1-NN | 70.4 / 19.3 / 30.3 | 87.5 / 39.7 / 54.6 | 80.1 / 52.0 / 63.0 |
| LDA | 65.7 / 32.7 / 43.7 | 84.8 / 55.5 / 67.1 | 78.7 / 64.4 / 70.8 |
| UUDAZS | 73.7 / 41.9 / 53.4 | 89.9 / 61.2 / 72.8 | 85.8 / 69.2 / 76.6 |
| MECA | 73.1 / 43.2 / 54.3 | 89.6 / 61.4 / 72.9 | 84.7 / 70.5 / 77.0 |
| ETN | 74.7 / 45.1 / 56.2 | 89.8 / 62.7 / 73.8 | 84.9 / 71.4 / 77.6 |
| BiDiLEL | 72.3 / 4.5 / 8.5 | 88.9 / 6.4 / 11.9 | 86.0 / 7.1 / 13.0 |
| LisGAN | 33.8 / 36.8 / 35.3 | 58.7 / 54.8 / 56.7 | 54.8 / 57.1 / 56.0 |
| **TUPL** | **68.1 / 53.0 / 59.6** | **87.5 / 75.3 / 80.9** | **84.6 / 75.0 / 81.5** |

*(Tables 3, 4, and 5 show similar consistent improvements for sources C, P, and R respectively, with TUPL achieving the highest Harmonic Means across almost all domain shifts.)*

### 4.3 Experimental results on Office31 dataset
TUPL achieves outstanding performances (e.g., 92.6%, 92.8%, 100.0% in H), outperforming other compared methods. ZSL methods like BiDiLEL fail in the GZSDA problem because they perform poorly on unseen classes.

**Table 6: Office31 (Selected Tasks)**
| Method | A→D (H) | A→W (H) | D→A (H) |
| :--- | :--- | :--- | :--- |
| 1-NN | 77.8 | 70.4 | 48.1 |
| UUDAZS | 87.4 | 81.5 | 67.1 |
| LisGAN | 30.4 | 25.1 | 47.8 |
| **TUPL** | **92.6** | **92.8** | **76.9** |

### 4.4 Experimental results on ImageCLEFDA dataset
TUPL ranks first with an average H accuracy of 88.6%, attaining remarkable performance boosts over generative methods like LisGAN (average H 52.0%). Adding the class center regularization term helps to learn discriminative representations effectively.

### 4.5 Algorithm analysis
#### 4.5.1 Ablation study
TUPL1 (without class center regularization term) shows dropped performances across all source-target pairs. t-SNE visualizations verify that adding the class center regularization term makes samples much more separable compared to just considering domain alignment.

#### 4.5.2 Importance of learning the networks $\phi$ and $\psi$
Utilizing the two networks simultaneously has a perfect effect on performance improvements. Directly transferring seen-unseen relationships from the source domain is inappropriate without modifications via source-target relationships.

#### 4.5.3 Parameter analysis
*   **Subspace dimension ($p$):** Performance is stable unless $p < 512$. Fixed at $p=1024$.
*   **Parameter $\alpha$:** Not very sensitive within the range of $[0.1, 10]$.

---

## 5 Conclusion and future work
We tackle the GZSDA task by learning target unseen class prototypes in the common subspace. For overcoming the domain shift, we project the source and target samples to the common subspace where the samples of the same class are near no matter which domains they are from. We preserve the discrimination of the subspace by making the samples near to their class prototypes and maintaining data variance. In addition, the target unseen class prototypes are learned simultaneously by source-target relationships and seen-unseen relationships. The experimental results show that the learned target unseen class prototypes are extremely useful for improving the GZSDA performance. 

In the future, we plan to use separate mappings from the source and target domains to the latent space for learning a better common subspace, and generate target unseen samples by GAN or VAE models.

---

## Acknowledgements
This work is supported by National Natural Science Foundation of China under (Grant Nos. 61806155, 62176197), China Postdoctoral Science Foundation funded project under Grant Nos. 2018M631125, National Natural Science Foundation of Shaanxi Province, Nature Science Foundation of Anhui Province, and Fundamental Research Funds for the Central Universities.

## Declarations
**Conflict of interest:** The authors declare that they have no conflict of interest.

---
*(Note: References 1-60 are omitted here for brevity but are present in the original manuscript document, detailing citations for Domain Adaptation, Zero-Shot Learning, GANs, VAEs, and datasets like Office-Home and ResNet50).*