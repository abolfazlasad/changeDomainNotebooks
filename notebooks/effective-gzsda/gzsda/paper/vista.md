**Cross-Domain Attribute Alignment with CLIP: A Rehearsal-Free Approach for Class-Incremental Unsupervised Domain Adaptation**

**Abstract**

Class-Incremental Unsupervised Domain Adaptation (CI-UDA) aims to adapt a model from a labeled source domain to an unlabeled target domain, where the sets of potential target classes appearing at different time steps are disjoint and are subsets of the source classes. The key to solving this problem lies in avoiding catastrophic forgetting of knowledge about previous target classes during continuously mitigating the domain shift. Most previous works cumberously combine two technical components. On one hand, they need to store and utilize rehearsal target sample from previous time steps to avoid catastrophic forgetting; on the other hand, they perform alignment only between classes shared across domains at each time step. Consequently, the memory will continuously increase and the asymmetric alignment may inevitably result in knowledge forgetting. In this paper, we propose to mine and preserve domain-invariant and class-agnostic knowledge to facilitate the CI-UDA task. Specifically, via using CLIP, we extract the class-agnostic properties which we name as “attribute”. In our framework, we learn a “key-value” pair to represent an attribute, where the key corresponds to the visual prototype and the value is the textual prompt. We maintain two attribute dictionaries, each corresponding to a different domain. Then we perform attribute alignment across domains to mitigate the domain shift, via encouraging visual attention consistency and prediction consistency. Through attribute modeling and cross-domain alignment, we effectively reduce catastrophic knowledge forgetting while mitigating the domain shift, in a rehearsal-free way. Experiments on three CI-UDA benchmarks demonstrate that our method outperforms previous state-of-the-art methods and effectively alleviates catastrophic forgetting. Code is available at [https://github.com/RyunMi/VisTA](https://github.com/RyunMi/VisTA).

**CCS Concepts**  
• Computing methodologies → Transfer learning; Lifelong machine learning.

**Keywords**  
Unsupervised Domain Adaptation, Class-Incremental Learning, CLIP.

**ACM Reference Format:**  
Kerun Mi, Guoliang Kang, Guangyu Li, Lin Zhao, Tao Zhou, and Chen Gong. 2025. Cross-Domain Attribute Alignment with CLIP: A Rehearsal-Free Approach for Class-Incremental Unsupervised Domain Adaptation. In *Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25)*, October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 10 pages. [https://doi.org/10.1145/3746027.3755184](https://doi.org/10.1145/3746027.3755184)

---



## 1 Introduction

Class-Incremental Learning (CIL) [2, 19, 47] aims to handle sequentially arriving tasks, where at each time step new classes emerge. The model needs to classify all seen classes during testing without access to the task ID. CIL methods generally rely on labeled data, which is often limited due to the high cost of data annotation in real-world scenarios [29, 31, 44, 45]. A feasible approach is to leverage an off-the-shelf labeled dataset (i.e., source domain) to transfer a model to a class-incremental unlabeled dataset (i.e., target domain), with the source domain containing all classes.

However, the distribution shift between domains poses significant challenges to the transferability of a model. Conventional unsupervised domain adaptation (UDA) or partial domain adaptation (PDA) methods can be utilized to mitigate the distribution shift by feature alignment [7, 11, 15, 18, 33, 36, 46] and domain-invariant knowledge transfer [24, 26, 30, 41, 42]. Nevertheless, existing domain adaptation methods may suffer from catastrophic knowledge forgetting [20] in class-incremental target domain, inspiring methods specifically designed for Class-Incremental Unsupervised Domain Adaptation (CI-UDA) [3, 17, 38].

Recently, several CI-UDA methods have been proposed [3, 17, 38]. They usually consist of two technical components. On one hand, they typically store and utilize rehearsal data from previous target classes (as illustrated by the yellow circles in Figure 1) to retain historical knowledge. However, rehearsal data may not be available due to constraints such as data privacy or memory limitations (e.g., the memory will increase as the number of tasks increases). On the other hand, to avoid negative knowledge transfer [10, 24], CI-UDA methods perform alignment only between classes shared across domains. However, they may still suffer from knowledge forgetting. Specifically, as shown in Figure 1, suppose the target class is “zebra” (a shared class) at step T. On one hand, the shared-class discovery process is imperfect. It may mistakenly treat private classes in source domain (i.e., source-private classes), such as “tiger” as a shared class, leading to misalignment. On the other hand, even in source-private classes, valuable knowledge exists but may be ignored during cross-domain alignment, such as “Black-White” of “panda,” and “Horse-like” of “horse,” which are also typical properties of “zebra” [9]. Recent works utilize CLIP-based [23] prompt learning [5, 8, 14, 22, 28, 49] to deal with domain adaptation problem. Technically, some methods can be directly applied to the CI-UDA setting, but as they typically do not consider the catastrophic forgetting issue, their performance is still far from satisfactory.

In this paper, we propose cross-domain Vision-Text Attribute Alignment (VisTA), a novel CI-UDA framework based on CLIP [23]. In our framework, we aim to mine and preserve domain-invariant and class-agnostic knowledge. Firstly, inspired by [35], we employ an Attribute Modeling module to utilize CLIP to extract class-agnostic properties which we refer to as “attribute”. We freeze the encoders of CLIP and construct a dictionary for source domain and target domain, respectively. The dictionaries store attributes in the form of “key-value” pairs, where the key and value bridge the visual and textual modalities. For each input image, several textual attributes are selected from the dictionary based on its visual attributes. These textual attributes, serving as prompts, are sent into CLIP to compute the class probability of the image. Then we perform attribute alignment across domains to mitigate the domain shift, via encouraging visual attention consistency and prediction consistency. Specifically, for each image in both domains, prompts are selected from the two dictionaries to compute paired class probabilities (one from each domain). However, since the two dictionaries are learned independently, the attributes selected from the other domain are domain-specific and may not effectively contribute to the current prediction due to the domain shift. Therefore, we introduce a Visual Attention Consistency (VAC) module to ensure the semantically similar attributes across domains are selected for paired prediction. VisTA then encourages Prediction Consistency by minimizing the Jensen–Shannon divergence between these paired probability distributions, enabling the learning of domain-invariant attributes. Benefiting from the modeling of domain-invariant and class-agnostic attributes, we are able to deal with the CI-UDA task in a rehearsal-free manner.

In a nutshell, the contributions of our work are summarized as follows:

- We propose a CI-UDA framework named VisTA, which leverages CLIP to learn class-agnostic attributes that act as prompts, achieving rehearsal-free training.
- VisTA learns domain-invariant attributes through attribute alignment, guided by a Visual Attention Consistency module and a Prediction Consistency loss.
- Extensive experiments firmly demonstrate the effectiveness of VisTA, as it achieves state-of-the-art performance on Office-31, Office-Home, and Mini-DomainNet.

---



## 2 Related Work

In this section, we review some relevant works, including unsupervised domain adaptation and vision language models.

**Unsupervised Domain Adaptation.** To mitigate distribution shift, conventional UDA methods or PDA methods typically fall into two main directions. The first direction of work aims to align the feature distributions across different domains. Common techniques include minimizing the statistical distribution metrics in the feature space directly [11, 33, 46] and applying adversarial learning to obtain domain-agnostic features [7, 15, 18]. The other direction of work seeks to transfer domain-invariant knowledge between models in source domain and target domain. For example, the works of [24, 26, 41] propose to learn an invariant classifier with consistent predictions, while [30, 42] propose to improve the performance of the target domain by knowledge distillation.

However, in practice, UDA is often integrated with continual learning problems [6, 34, 39]. One scenario is where target data arrives in a streaming manner with different classes. In this scenario, conventional UDA methods suffer from the catastrophic forgetting problem [20]. Therefore, some recent CI-UDA methods [3, 17, 38] have been developed to mitigate the domain shift while learning class-incremental target classes. For instance, ProCA [17] detects the shared classes by computing cumulative prediction probabilities of target examples and achieves adaptation through prototype alignment. PLDCA [38] builds upon ProCA and further alleviates negative transfer through domain-level and instance-level contrastive alignment. Besides, CROTO [3] designs a multi-granularity class prototype self-organization module and a prototype topology distillation module to handle CI-UDA in a source-free scenario (i.e., CI-SFUDA). It is worth noting that existing CI-UDA methods are suboptimal owing to continuously increasing memory and asymmetric alignment.

**Vision Language Models.** Recent Vision Language Models (VLMs) like CLIP [23] have demonstrated impressive performance on various downstream vision tasks by pretraining on large-scale image-text pairs [43]. VLMs typically use hand-crafted text like “a photo of a [class name]” for zero-shot prediction on downstream tasks, which preserves generalization knowledge while maintaining low computational cost. However, hand-crafted text is not always effective, and thus prompt learning has gained increasing attention. For instance, CoOp [49] and CoCoOp [48] use learnable continuous prompts to improve the generalization performance of CLIP. MaPL [12] proposes multi-modal prompt learning to align text and image representations. Moreover, since only the prompts should be stored, certain prompt learning methods serve as rehearsal-free learners, which can be effectively utilized to address class-incremental problem [35].

However, these prompt learning methods often suffer from performance degradation when encountering domain shift problems. To address this, DAPL [8] introduces domain-specific and domain-agnostic prompts to learn the label distribution of target domain. AD-CLIP [28] learns domain-invariant prompts by combining domain style information with image content information. DAMP [5] mutually aligns visual and textual embeddings to learn domain-agnostic prompts. PGA [22] frames UDA as a multi-objective optimization problem and promotes consensus among per-objective gradients. Although existing prompt learning methods have shown quite promising performance, they cannot deal with CI-UDA, as the historical knowledge encoded in prompts may be overwritten by new class information, leading to catastrophic forgetting. To this end, in this paper, we propose a rehearsal-free method based on prompt learning for CI-UDA, which effectively reduces catastrophic forgetting while mitigating distribution shift.

---



## 3 Preliminaries



### 3.1 CI-UDA Problem Formulation

In CI-UDA, we consider two domains, including the labeled source domain D_s = (x_i^s, y_i^s)*{i=1}^{n_s} (where x_i^s denotes the i-th source image, y_i^s \in 1, 2, \dots, C denotes the label of i-th image and n_s means the number of source examples), and unlabeled target domain D_t = x_i^t*{i=1}^{n_t}. The sample in D_s is available at all time steps and covers all the considered C classes, and the sample in D_t comes incrementally. For each time step, the underlying class set of D_t is only a subset of 1, \dots, C and the class sets of different time steps are disjoint.

The goal of CI-UDA is to learn a model by leveraging data from D_s and class-incremental D_t, such that it performs well on all seen classes of D_t during testing.

### 3.2 Prompt Learning in CLIP

CLIP [23], a prominent VLM, pretrains an image encoder E_I and a text encoder E_T on large-scale image-text pairs to learn well-aligned visual and textual representations. In downstream tasks, class-specific textual prompts P_k may be utilized for each class k (e.g., “a photo of [CLS_k]”, where [CLS_k] is the k-th class name). CLIP predicts the probability that an input image x belongs to each class by computing the cosine similarity between the visual feature z = E_I(x) \in \mathbb{R}^D (where D is the feature dimension) and the class-wise text embeddings w_k = E_T(P_k) \in \mathbb{R}^D, k = 1, \dots, C:


p(y = k \mid x) = \frac{\exp(\cos(w_k, z)/\tau)}{\sum_{c=1}^{C} \exp(\cos(w_c, z)/\tau)},


where \cos(\cdot, \cdot) is cosine similarity, and \tau is temperature parameter.

However, such hand-crafted prompts may be suboptimal. To further enhance the performance of CLIP in downstream tasks, CoOp [49] introduces learnable continuous vectors V with length M to replace hand-crafted prompt templates. The learnable prompt for class k is then defined as P_k = [V_1, V_2, \cdots, V_M, \text{CLS}_k]. During training, the prompt P_k is updated to minimize the cross-entropy loss on sample from downstream tasks. For the inference, they follow the same way which utilizes Eq. (1) to predict the label for each example. The only difference is that they use the learned prompts instead of manually designed prompts for prediction.

---



## 4 Method

As shown in Figure 2, at time step T, VisTA processes D_s and the current D_t through E_I, generating visual features z^s and z^t. Taking z^t as an example, we retrieve L textual attributes from target dictionary based on the cosine similarity between z^t and visual attributes (Section 4.1). These attributes serve as textual prompts for E_T. Concurrently, we propose a Visual Attention Consistency module (Section 4.2), which applies a Grad-CAM-based attention heatmap matching mechanism to select L source attributes with similar semantic concepts as prompts. This process yields two class probability distributions for z^t. The target attributes then are optimized through a Prediction Consistency loss (Section 4.2) to enable the learning of class-agnostic and domain-invariant knowledge. A similar procedure is adopted for z^s. Finally, Section 4.3 discusses two regularization terms and the training objective of VisTA.

### 4.1 Attribute Modeling

In CI-UDA setting, the underlying class set of target domain at each time step is only a subset of that of source domain and is disjoint from that of previous time steps. If mitigating knowledge forgetting at the class-level, we may inevitably need to store previous target sample and discover shared classes between domains at each time step to perform alignment—a cumbersome process that may introduce too much noise during training. In this paper, we aim to mitigate the knowledge forgetting in CI-UDA at the “attribute” level. The “attribute” refers to the basic components which combine to support correct predictions.

Specifically, with CLIP extracting visual and textual features, we represent each attribute as a ‘key-value’ pair, where the value refers to the textual representation of attribute and the key refers to the visual representation of attribute, in other words, the visual prototype. Formally, attributes are denoted as:


[\mathcal{K}, \mathcal{A}] := [k_1, a_1], k_2, a_2, \dots, k_N, a_N],


where each key k_i \in \mathbb{R}^D (i = 1, 2, \dots, N) is designed to capture the visual attributes of an image x, and each value a_i \in \mathbb{R}^{M \times D} (i = 1, 2, \dots, N) encodes the textual description of a specific attribute.

VisTA maintains a source attribute dictionary [\mathcal{K}^s, \mathcal{A}^s] and a target attribute dictionary [\mathcal{K}^t, \mathcal{A}^t]. We design specific update strategies for attributes to learn class-agnostic knowledge.

**Visual Attributes.** We perform k-means++clustering [1] on all source features to obtain source visual attributes \mathcal{K}^s before training and keep \mathcal{K}^s fixed during training. As the target classes appearing at different time steps are disjoint in CI-UDA, target visual attributes \mathcal{K}^t are initialized via k-means++ clustering on the data from the first time step in the class-incremental training sequences. During each subsequent time step in class-incremental learning process, we apply a moving average strategy to update \mathcal{K}^t.

**Textual Attributes.** VisTA randomly initializes \mathcal{A}^s and \mathcal{A}^t. These textual attributes are then modeled through supervised training or self-training to mine and preserve class-agnostic knowledge.

Given an image x from both domains, we select the top-L visual attributes \tilde{\mathcal{K}} \subseteq \mathcal{K} based on their cosine similarity to x. The paired textual attributes are then indexed from the dictionary as \tilde{\mathcal{A}} = \tilde{a}_{1:L}. These textual attributes are concatenated with the class name to form the prompt:


P_k(\tilde{\mathcal{A}}) = [\tilde{a}_1, \tilde{a}_2, \dots, \tilde{a}_L, \text{CLS}_k], \quad k = 1, \dots, C.


The source textual attributes \mathcal{A}^s are optimized by minimizing cross-entropy loss on labeled D_s:


\mathcal{L}_{\text{sup}}^s = -\log p(y = y^s \mid x^s).


Moving to unlabeled D_t, target textual attributes \mathcal{A}^t are optimized by minimizing self-training loss:


\mathcal{L}_{\text{sup}}^t = -\mathbb{I}(\max(\hat{p}) \ge \gamma) \log p(y = \hat{y}^t \mid x^t),


where \hat{y}^t represents the pseudo-label, \mathbb{I}(\cdot) is the indicator function, \gamma is a threshold to select high-confidence pseudo-label of target example, and \hat{p} denotes the debiased soft pseudo-label. This debiasing follows DebiasPL [37], which aims to enhance the reliability of pseudo-labels and has recently been adopted in several CLIP-based UDA methods [13, 16]. In detail, \hat{p} is computed as:


\hat{p} = p - \tau \log q, \quad q \leftarrow m q + (1-m) \frac{1}{B} \sum_{j=1}^{B} p_j,


where m is a momentum coefficient, \tau is a de bias factor, B denotes the batch size, and q is initialized before training as a uniform probability vector over C classes.

### 4.2 Cross-Domain Attribute Alignment

To mitigate the domain shift in CI-UDA, VisTA performs cross-domain attribute alignment through a Visual Attention Consistency module and a Prediction Consistency loss. We want to mention that previous work AttriCLIP [35] can also extract class-agnostic attributes to benefit general continual learning, but cannot guarantee the domain-invariant property, which is crucial for CI-UDA. Hence, we design two novel modules to address this limitation.

**Visual Attention Consistency (VAC).** Taking x^t as an example, we select L visual attributes \mathcal{A}^t from target dictionary through cosine similarity between x^t and \mathcal{K}^t. Then we select L source visual attributes \mathcal{A}^s for alignment. Note that the update of visual attributes mentioned previously (Section 4.1) is domain-specific, so selecting \mathcal{A}^s for x^t by cosine similarity may introduce bias due to domain shift. Therefore, based on Grad-CAM [27], VisTA proposes an attention heatmap matching mechanism for cross-domain attribute selection.

Specifically, we compute L target CAM scores for x^t as:


S_{\text{CAM}}^t = \frac{\exp(\cos(E_T(a_m^t), z^t)/\tau)}{\sum_{n=1}^{L} \exp(\cos(E_T(a_n^t), z^t)/\tau)},


where a_m^t \in \mathcal{A}^t, (m = 1, \dots, L) are individual textual attributes.

Then we use the gradients of S_{\text{CAM}}^t with respect to the features from a layer as weights, and perform a weighted aggregation to highlight attribute-level discriminative regions. This procedure follows Grad-CAM to generate L heatmaps H^t for x^t. In the same way, we also compute N source CAM scores S_{\text{CAM}}^s for all source attributes \mathcal{A}^s, thereby obtaining N candidate heatmaps H^s for x^t.

To select \mathcal{A}^s from N candidates, VisTA quantify the visual attention consistency between flattened H^t and H^s using the Pearson Correlation Coefficient (PCC): \rho = \frac{\text{cov}(H^s, H^t)}{\sigma_{H^s} \sigma_{H^t}} \in [-1, 1], where cov represents the covariance, and \sigma denotes the standard deviation. A higher value of \rho highlights attributes for prioritized selection.

The visual attention consistency in images can be interpreted as the similarity of semantic concepts. As illustrated in the bottom-right panel of Figure 2, we analyze a “panda” image from D_t. The attention heatmap H^t of the selected target attribute \mathcal{A}^t exhibits concentrated activations in the central-right region of the image, likely corresponding to the semantic concept of “Head”. Notably, the selected source attribute \mathcal{A}^s with the highest \rho = 0.66 explicitly aligns with the same semantic concept of “Head”.

Therefore, for L selected target attributes \mathcal{A}^t and total N source attributes \mathcal{A}^s, we compute L \times N PCC \rho. Leveraging these values, we match and identify the top-L attributes \mathcal{A}^s as the cross-domain attribute selection result for x^t. The procedure to employ VAC module is identical for x^s. Importantly, PCC, defined as cosine similarity on normalized vectors, reduces the influence of global style and is thus suitable for quantifying the visual attention consistency.

**Prediction Consistency.** To achieve attribute alignment, we introduce a Prediction Consistency loss applied to the attributes selected by the VAC module, which enforces domain invariance for these attributes exhibiting semantic similarity.

As illustrated in the top-right panel of Figure 2, the selected attributes \mathcal{A}^s and \mathcal{A}^t are served as textual prompts for E_T to generate class-wise text embeddings e_k^{s,t} = E_T(P_k(\mathcal{A}^{s,t})), k = 1, \dots, C. This enables the computation of paired class probabilities for x using Equation (1). For example, given an image x^t is sampled from D_t, we utilize prompts P(\mathcal{A}^s) and P(\mathcal{A}^t) to obtain class probability vectors p^{ts} and p^{tt}, respectively. For an image x^s is sampled from D_s, analogous terms p^{ss} and p^{st} are computed. Here, the superscripts of p denote the domain of the x (first symbol) and the dictionary from which the prompt is selected (second symbol).

The Prediction Consistency loss is achieved by minimizing the Jensen–Shannon divergence D_{JS} between each pair of class probabilities:


\mathcal{L}*{\text{con}} = D*{JS}(p^{ss}, p^{st}) + D_{JS}(p^{tt}, p^{ts}).


In this way, VisTA effectively reduces catastrophic knowledge forgetting while mitigating the domain shift by learning class-agnostic and domain-invariant attributes. During inference, we use p^{tt} as the prediction score of target example.

### 4.3 Training Objective

Notably, we aim to enhance the generalization capacity of \mathcal{A}^s, enabling it to effectively guide predictions in D_t. Inspired by [40], VisTA proposes a regularization loss which minimizes the distance between class-wise embeddings e_k^s generated by \mathcal{A}^s and those derived from hand-crafted prompts (w_k):


\mathcal{L}*{\text{hp}} = \sum*{k=1}^{C} |e_k^s - w_k|.


Finally, to promote the diversity of textual attributes, a regularization loss is applied separately to both D_s and D_t to enforce orthogonality among the attributes within \mathcal{A}:


\mathcal{L}*{\text{div}} = \frac{1}{N(N-1)} \sum*{m=1}^{N} \sum_{n=m+1}^{N} |\cos\langle E_T(a_m), E_T(a_n)\rangle|.


As a result, the final optimization objective of VisTA is:


\mathcal{L} = \mathcal{L}*{\text{sup}} + \lambda_1 \mathcal{L}*{\text{con}} + \lambda_2 \mathcal{L}*{\text{hp}} + \lambda_3 \mathcal{L}*{\text{div}},


where \lambda_1, \lambda_2, and \lambda_3 are non-negative trade-off weights, and \mathcal{L}*{\text{sup}} = \mathcal{L}*{\text{sup}}^s + \mathcal{L}_{\text{sup}}^t is the classification loss.

---



## 5 Experiment



### 5.1 Experimental Setup

**Datasets.** Office-31 [25] includes 31 categories from three domains: Amazon (A), DSLR (D), Webcam (W), totaling 4,600 images. Office-Home [32] comprises 65 categories across four distinct domains: Art (A), Clipart (C), Product (P), and Real World (R), totaling 15,500 images. Mini-DomainNet is a subset of DomainNet [21] and includes 126 categories across four domains: Clipart (C), Painting (P), Real World (R), and Sketch (S).

Following ProCA [17], we divide each domain of Office-31 into three disjoint subsets, each containing 10 classes in alphabetical order, and divide each domain of Office-Home into six disjoint subsets, each containing 10 classes in an order consistent with ProCA [17]. Additionally, as the first CI-UDA method to handle Mini-DomainNet, we divide each domain into six disjoint subsets, each containing 20 classes in alphabetical order. More details of datasets construction are in Appendix A.

**Baseline Methods.** We compare VisTA with five types of methods: (1) source-only: ViT-B/16 and CoOp [49]; (2) zero-shot: CLIP; (3) existing CI-UDA methods: ProCA [17] and PLDCA [38]; (4) prompt learning method for CIL: AttriCLIP [28]; (5) prompt learning methods for UDA: AD-CLIP [28], DAMP [5], PGA [22], and DAPL [8].

**Implementation Details.** We use ViT-B/16 [4] as the image encoder for VisTA and all baseline methods. Details on the hyperparameters of VisTA, as well as the training procedures for VisTA and several baseline methods, are provided in Appendix B. We analyze the sensitivity of our method to hyperparameters in Section 5.3.

**Evaluation Metrics.** To comprehensively evaluate the performance of VisTA, we employ three metrics for CI-UDA:

1. **Final Accuracy**: the classification accuracy across all classes at the final time step for each adaptation task;
2. **Step-level Accuracy**: the average classification accuracy over all adaptation tasks at each time step;
3. **S-1 Accuracy**: the average classification accuracy of all adaptation tasks at each time step for classes in Step-1.



### 5.2 Comparisons with previous state-of-the-arts

The Final Accuracy results are summarized in Tables 1, 2, 3, while the Step-level Accuracy results are detailed in Table 4. The numbers reported in the tables are reproduced by us using the officially released code, unless otherwise specified. Additionally, the results of S-1 Accuracy are visualized in Figure 3. Due to page limitations, complete results with extended details for all metrics across three benchmarks are reported in Appendix C.

**Comprehensive learning ability of VisTA.** In Tables 1, 2, 3, each column corresponds to one specific CI-UDA task, i.e., Source → Target. The column “Avg.” means the average of Final Accuracy for all CI-UDA tasks.

The results of Final Accuracy demonstrate that VisTA performs favorably against other methods. VisTA achieves improvements of 0.7%, 8.7%, and 24.2% over the leading CI-UDA method PLDCA on Office-31, Office-Home, and Mini-DomainNet, respectively. It also outperforms other top competitors, surpassing CoOp by 0.3% on Office-31, AD-CLIP by 1.5% on Office-Home, and DAPL by 1.7% on Mini-DomainNet. We find that the performance advantages of VisTA scale with benchmark complexity (Office-31→Office-Home→Mini-DomainNet), as quantified by the number of time steps and underlying classes. It substantiates that the class-agnostic and domain-invariant attributes learned by VisTA effectively alleviate catastrophic forgetting and domain shift, especially in scenarios with numerous classes and long-sequence tasks.

The results further indicate that existing CI-UDA methods with ViT-B/16 underperform against source-only CoOp (on all benchmarks) and zero-shot CLIP (except on Office-31). This highlights the exceptional generalization ability of pretrained CLIP, which encodes comprehensive category knowledge via prompt learning.

Additionally, some UDA methods (i.e., DAMP, PGA) and CIL method (i.e., AttriCLIP) based on prompt learning obtain worse results than source-only CoOp across all three benchmarks. This suggests that an exclusive focus on mitigating either domain shift or catastrophic forgetting may reduce the generalization capability of prompt learning methods in handling CI-UDA.

**Incremental learning ability of VisTA.** The Step-level Accuracy and S-1 Accuracy are used to evaluate whether a method can retain knowledge of previous target classes while learning new ones. Each column in Table 4 represents the average accuracy of all adaptation tasks at a specific time step, and the column “Avg.” denotes the average Step-level Accuracy. The x-axis indicates time steps and the y-axis shows the S-1 Accuracy in Figure 3(a).

It is observed that all comparison methods are affected by catastrophic forgetting, resulting in lower S-1 Accuracy at the final step compared with the first step. Some methods also exhibit a decline in Step-level Accuracy over time, indicating that they not only forget previously learned target classes but also struggle to learn new ones. In contrast, VisTA shows an upward trend in both Step-level Accuracy and S-1 Accuracy over time, achieving the best performance at both the final step and on average. The results of both metrics also show that the performance in the early adaptation phase is not state-of-the-art. We analyze that although VisTA aligns attributes across domains as much as possible, the learned \mathcal{A}^t is not sufficiently refined in the early stage, while \mathcal{A}^s is trained on all classes. Aligning \mathcal{A}^s and \mathcal{A}^t with unequal learning progress may harm performance. Only after \mathcal{A}^t fully learns the attributes over time steps can the performance advantages manifest.

Moreover, Figure 3 (b) illustrates the percentage change in S-1 Accuracy at each step compared with the first step. VisTA is the only method that consistently achieves positive gains and demonstrates continuous improvement. This indicates that VisTA effectively preserves and progressively reinforces knowledge of class-incremental D_t when addressing CI-UDA on Office-Home.

**Extension to source-free scenario.** In Appendix D, we compare VisTA with the CI-SFUDA method GROTO [3]. These results demonstrate the capabilities of VisTA under source-free scenarios.

### 5.3 Ablation Analysis

Table 5 presents Final Accuracy and S-1 Accuracy at the final step (i.e., Final S-1 Accuracy) on Office-Home, obtained by removing specific modules (i.e., “w/o.”) while keeping other settings identical. More studies about various VLMs and computational overhead are reported in Appendix E.

**Effect of VAC.** The “w/o. VAC” is achieved by removing the VAC module, with attributes selected directly from the dictionary corresponding to the other domain using cosine similarity. This leads to a noticeable performance degradation, demonstrating that the selection enabled by the VAC module effectively mitigates bias caused by the domain shift.

**Effect of \mathcal{L}_{\text{con}}.** The “w/o. \mathcal{L}*{\text{con}}” indicates that D_s and D_t share the same attributes dictionary without \mathcal{L}*{\text{con}}. Notably, the “w/o. \mathcal{L}_{\text{con}}” leads to the sharpest performance decline, demonstrating that separate attribute modeling in D_s, D_t with alignment effectively prevents conflicting knowledge acquisition in entangled attributes.

**Effect of regularization terms.** The “w/o. \mathcal{L}*{\text{hp}}” and “w/o. \mathcal{L}*{\text{div}}” denote the \mathcal{L}*{\text{hp}} and \mathcal{L}*{\text{div}} are removed from the objective (11), respectively. The observed performance decline confirms the effectiveness of both regularization terms in attribute learning.

**Sensitivity to loss weights.** We need to determine three loss weights of the objective (11). Empirical observations reveal \lambda_3 has negligible impact, so we fix it at 1.0 and vary \lambda_1 and \lambda_2. As shown in Figure 4(a), the performance of VisTA is generally insensitive to \lambda_1, \lambda_2 \in [5, 10, 15, 20, 25], with best performance at \lambda_1 = 25, \lambda_2 = 20.

**Sensitivity to hyperparameters of attribute dictionary.** We consider the following hyperparameters, such as the prompt length M, the number of attributes in the bank N, and the number of selected attributes L. To cap training and computational costs, we fix N = 8 and explore variations in M and L. Figure 4(b) shows that the performance of VisTA is robust to M. Moreover, when a sufficient number of attributes are selected (L \ge 2), VisTA also exhibits robustness to L.

**Visualization of textual attributes.** To verify whether the learned attributes reflect the semantic concept of images, we visualize the image contents of distinct classes corresponding to different attributes using Grad-CAM [27]. To further demonstrate the attribute matching process in VAC module, we present examples of target classes “horse,” “panda,” and “zebra” from Mini-DomainNet. As shown in Figure 5, the learned \mathcal{A}^t exhibit two key properties: (1) different attributes reflect distinct semantic concepts within the same image (e.g., \mathcal{A}^t_2 \rightarrow “Background,” \mathcal{A}^t_4 \rightarrow “Head,” \mathcal{A}^t_8 \rightarrow “Body”), and (2) the same attribute reflects identical semantic concept across different images. This demonstrates that the learned \mathcal{A}^t are class-agnostic and diverse, effectively retaining knowledge to alleviate catastrophic forgetting. Unlike the learned \mathcal{A}^t corresponding to D_t, the \mathcal{A}^s, affected by the distribution shift, fail to learn attributes identical to \mathcal{A}^t and do not exhibit property (2). However, \mathcal{A}^s may still be partially similar to \mathcal{A}^t in semantic concepts. Building on this similarity, VAC module selects cross-domain attributes through a \rho-guided matching mechanism to learn domain-invariant attributes that mitigate the distribution shift.

**Visualization of visual attributes.** To verify whether \mathcal{K}^s and \mathcal{K}^t can adequately cover the attributes of all examples, we conduct t-SNE visualizations for C → P task from Office-Home at steps 1, 4, and 6 as shown in Figure 6. It displays CLIP-extracted visual features from D_s (orange) and D_t (green), along with \mathcal{K} obtained through k-means++ clustering. We observe that the elements within \mathcal{K}^s (blue) and \mathcal{K}^t (red) consistently remain diverse and distinct. Furthermore, \mathcal{K}^t at the final step successfully covers the examples from other time steps (gray), demonstrating that \mathcal{K} effectively captures the overall attributes of the sample from both domains.

---



## Conclusion

In this paper, we propose to model and align attributes across domains based on CLIP to deal with the class-incremental unsupervised domain adaptation (CI-UDA), which is a rehearsal-free approach. Specifically, via CLIP, we extract the class-agnostic properties, i.e., attributes. Each attribute is represented as a “key-value” pair where the key corresponds to visual prototype and the value corresponds to textual prompt. In our method, we learn to construct two dictionaries, each corresponding to a specific domain. Each dictionary consists of a group of attributes. Then we perform attribute alignment to make attribute invariant across domains via utilizing the consistency knowledge including visual attention consistency and prediction consistency. Experiments on three benchmarks verify the effectiveness of our proposed method.

---

**Acknowledgments**

This research is supported by the National Natural Science Foundation of China (NSFC) under Grants Nos. 62336003, 12371510, 92370114, and 62006119; and the National Key Research and Development Program of China (International Collaboration Special Project, No. SQ2023YFE0102775).