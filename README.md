# Tunix_hackathon
A repo for my participation in the Tunix hackathon

My goal here was to get familiarized with Tunix and the overall State of the Art (by December 2025) on fine tuning LLMs.

To get a good idea of what the industry is doing to fine tune LLM's I highly advise you to read the  ([deepseek paper](https://arxiv.org/pdf/2501.12948)). I would only recommend this if you are somewhat familiar with Reinforcement Learning, more specifically if you know what PPO is.

The goal of the ([Google Tunix hackaton](https://www.kaggle.com/competitions/google-tunix-hackathon)) is to 'train a model to show its work by laying out a reasoning trace before landing on an answer' using their new Tunix library, which is built on top of Jax to facilitate LLM post training.

Tunix supports a few ways of fine tuning your LLM, the ones I was interested in diving deeper into are:

* Supervised fine Tuning (SFT)
* Reinforcement Learning


### Supervised Fine Tuning

SFT consists in carrying on the training of your LLM on labelled data, the goal is for it to learn to perform certain tasks you want to excel in. This is usually the first step LLM engineers take before resorting to RLHF.

Tunix supports SFT both via full model fine tuning, which consumes a lot of resources, or Parameter Efficient Fine Tuning. 

PEFT is a whole world of methods, and it is beyond the scope of this project to get into it. That is why I will stick to the most popular PEFT method of current days, LoRA.

One should use PEFT when there is a need to customize your raw LLM, one is in posession of groud truth labells for a specific case and when one does not wish to optimize according to human preference (for that we use RLHF).


### Reiforcement Learning: Group-relative policy optimization (GRPO)

GRPO is a sort of PPO without value function estimation. I kinda try to see it as PPO adapted solely for LLM's. SO while PPO needs a value function to estimate advantages; GRPO computes a relative advantage across a batch/group of 'trajectories' (in the case of LLM's possible answers to a question), avoiding critic errors. I also read somewhere that 'GRPO is especially useful in tasks with sparse or noisy rewards', but I saw no explanation as to why, maybe because we take batches of possible 'trajectories' and compute their advantage... idk, I wouldn't be so sure about it.


Lets dive into the juicy details, I'll keep here some general approaches for GRPO, knowing that Deepseek implemented a variant of this overall formulation.

The GRPO objective function is a **clipped surrogate**, similar to PPO:


$$
L^{GRPO}_{\text{total}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \min \Big( r_i(\theta) \hat{A}_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \Big)
- \beta \cdot \frac{1}{N} \sum_{i=1}^{N} \text{KL}\big[\pi_{\text{old}}(\cdot|s_i) \,\|\, \pi_\theta(\cdot|s_i)\big]
$$

$$
\text{with } \hat{A}_i = \frac{R_i - \frac{1}{N} \sum_{j=1}^{N} R_j}{\sqrt{\frac{1}{N} \sum_{j=1}^{N} \Big(R_j - \frac{1}{N} \sum_{k=1}^{N} R_k \Big)^2} + \delta}
$$

Where:

- \( $\beta$) : KL-penalty coefficient (tunes how strongly to penalize deviation from old policy).  
- \( $\text{KL}[\pi_{\text{old}} \| \pi_\theta]$ \) : KL-divergence between old and new policy for state \( s_i \).  


- \( $\theta$ \) : Policy parameters.  
- \( i \) : Index over trajectories in the sampled group.  
- \( $r_i(\theta) = \frac{\pi_\theta(a_i | s_i)}{\pi_{\text{old}}(a_i | s_i)}$ \) : Probability ratio of new vs old policy.  
- \( $\hat{A}_i$ \) : **Relative advantage** of trajectory \( i \), computed as:

- \( $R_i$ \) : Observed reward of trajectory \( i \).  
- \( N \) : Number of trajectories in the group.  
- \( $\delta$ \) : Small constant for numerical stability.  
- \( $\epsilon$ \) : Clipping hyperparameter (same as in PPO).  

And where, from what I understood : 

1. The **group-based normalization** removes the need for a critic/value network.  
2. The **clip** ensures policy updates are stable.  
3. It is simpler and easier to implement than PPO
4. GRPO is on-policy, it samples data straight from the model(policy) being optimized
5. GRPO’s sampling is online because trajectories are generated on-the-fly by the current policy interacting with the environment (or model), as opposed to training on a fixed offline dataset.
