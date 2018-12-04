# ML Projects

A central repository for several projects I worked on during my Masters program (Masters of Computational Data Science) at Carnegie Mellon University's Language Technologies Institute (CMU LTI).

For each project, I show a writeup (some drafted using ML conference templates) of the results of the project, as well as the code.

## Tree Re-Ranker for Question Answering

* Context: MCDS Capstone Project under advisor Eric Nyberg
* Course Grade: A+
* Project Name: `squad_tree_ranker`
* My Contribution: main contributor (team of 3).  Other two worked primarily on different model.
* Description: I developed a novel tree-based algorithm to predict answer spans by ranking syntactic constituents.  I had hypothesized that ranking syntactic units, as opposed to predicting span indices with a pointer network, would ensure answer coherence, encourage phrase-level attention, and create a soft learning objective aligned with the F1 evaluation metric.  I used a Tree-LSTM to recursively encode syntax tree nodes.  Then, a co-attention layer fused question and passage before a bi-LSTM pre-order tree traversal modelled dependencies.  The model scored each syntactic unit and maximized the expected F1 score during training.  I added added an ensembled answer verifier to handle unanswerable questions. The single-model system now achieves an official F1 evaluation score of 62.3 on the competitive Stanford Question Answering Dataset 2.0 (SQUAD).
* Notes: As of December 2018, the model ranks 24th best on the competitive [leaderboard](https://rajpurkar.github.io/SQuAD-explorer/).

## Resolving Implicit Coordination in Multi-Agent Deep Reinforcement Learning with Deep Q-Networks & Game Theory

* Context: Final Project for "Deep Reinforcement Learning & Control" class taught by Ruslan Salakhutdinov
* Course Grade: A
* Project Name: `multi_agent_reinforcement_learning`
* My Contribution: equal contributor (team of 3)
* Description:  For a class on Deep Reinforcement Learning, we explored multi-agent cooperative environments, realistic settings where robots share a common goal. To manage state space explosion, we combined Nash game theory for independent policy selection with Deep Q-Networks and applied the model to 3 custom environments in OpenAI Gym.

## Latent Sentence Level Representation Learning

* Context: Final Project for "Neural Networks for NLP" class taught by Graham Neubig
* Course Grade: A
* Project Name: `latent_sentence_structure`
* My Contribution: equal contributor (team of 3)
* Description: Our team reproduced the Stack-augmented Parser-Interpreter Neural Network (SPINN) created by the Stanford NLP team. I then explored unsupervised training objectives which relied solely on a downstream semantic objective, such as the SNLI entailment task. I found that task-specific structure learning does not materially improve performance, but can be effective at sentence understanding in the absence of labeled parse data. 

## Modeling Visual Question Answering as Hard Attention over Semantic Image Regions (Code Unavailable)

* Context: Final Project for "Advanced Multimodal Machine Learning" class taught by Louis-Philippe Morency
* Course Grade: A+
* Project Name: `visual_question_answering`
* My Contribution: equal contributor (team of 4)
* Description: We incorporate hard attention over image regions using reinforcement learning, which allows the model to focus on a single ‘glimpse’ at a time in a recurrent manner.  Secondly, we leverage semantic image segmentation maps, which produce pixel-wise semantic labels, to extract glimpse features in the form of semantic histograms.  While not delivering state of the art performance, our approach is guided by intuitive and grounded principles we hope can motivate future research.
