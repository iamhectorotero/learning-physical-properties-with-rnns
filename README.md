# Learning physical properties with RNNs
![](https://travis-ci.com/iamhectorotero/learning-physical-properties-with-rnns.svg?token=vGXQzmA3wxTt9C2pBBg4&branch=master)
![](https://codecov.io/gh/iamhectorotero/learning-physical-properties-with-rnns/branch/master/graphs/badge.svg?token=krWcTqni7k)


Human beings possess an innate understanding of physics. We can, for example, predict that an object will fall just by **observing** how it's standing on a table or learn a ball's weight by passing it from hand to hand. 'When' these abilities are acquired and 'how' our brain performs are a matter of discussion among scientists. [Bramley et al., 2018] developed an environment to better assess our abilities at guessing these hidden properties. Using the environment shown below, participants were tasked with guessing the mass of the labelled pucks and the attraction or repulsion forces acting between them.  

![A passive trial](passive_trial.gif)

In this work, we trained Recurrent Neural Networks on the same task and show that not only do they achieve a close (but higher) accuracy to human participants but their answering patterns and certainty also resemble those of the participants. This is not trivial as the Ideal Observer approach developed by [Bramley et al., 2018], a model with access to the underlying simulation model, did not manage to capture their behavior as precisely.
 

# Table of contents
1. [The environment and task](#the-environment-and-task)
2. [Libraries](#libraries)
3. [Notebooks](#notebooks)
4. [Installation Guide](#installation-guide)
5. [References](#references)


# The environment and task
We are presented with an environment that includes four pucks that move according to Newtonian physics. We are tasked with answering questions regarding the **target pucks** (labelled A and B). The remaining distractor pucks (painted blue) are there just to complicate the task (they can attract/repel the target pucks and provoke collisions).

Two types of questions are asked:

- Which puck is heavier? 'A', 'B' or do they weigh the 'same'?
- Do the pucks 'attract', 'repel' or 'no force' acts between them?

# Libraries

- `simulator`: includes the code necessary to generate the physical environment and generate passive trials or run active simulations.  Its main configuration can be checked in `environment.py`.
- `isaac`: all the tools necessary to generate datasets with passive simulations, postprocess them and train Recurrent Neural Networks to predict the environment's latent physical properties (mass or force). Also includes code to evaluate the resulting models and visualize trials. **Unit tests** for the code in this library are incldued in `tests/isaac_tests`. ![](https://travis-ci.com/iamhectorotero/learning-physical-properties-with-rnns.svg?token=vGXQzmA3wxTt9C2pBBg4&branch=master) ![](https://codecov.io/gh/iamhectorotero/learning-physical-properties-with-rnns/branch/master/graphs/badge.svg?token=krWcTqni7k)

# Notebooks
### cogsci_experiments
Notebooks spawning from the experiments performed for CogSci 2020 and their corresponding visualization. These include:
##### Passive
- RNN cell type selection. (Notebooks 0 and 0b).
- Single branch vs Multi-branch neural network experiments (Notebooks 1a-1e).
- Feature selection (Notebooks 2a and 2b).
- Sequence length and resolution selection. (Notebooks 3a and 3b).
- Training the model for 300 epochs (Notebooks 4a and 4b).
- Learning rate experiments (Notebooks 5a and 5b).
- Magnitude and angle experiments (Notebook 6a and 6b. 
- Training 25 multibranch models. (Notebook 7a, 7b and 7c).
- Comparison between participants and RNN models. (Notebooks 9).
- Is the RNN softmax output a good predictor of human guesses? (Notebook 10).
- Predictive Divergence vs RNN Interval Information. 

# Installation guide

To run the notebooks and/or libraries in this repository, it is necessary to install some dependencies.
The file `conda_environment.yml` lists these dependencies and can be directly used to create a
Conda environment with them installed (see [HERE](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
for a tutorial on how to install Conda). Once Conda is available, run:
```
conda env create -f conda_environment.yml
conda activate learning-physical-properties-with-rnns
pip install ./libraries
```
These commands will first create an environment named "diss", then activate it and install in it the
libraries in the repository ("isaac", "simulator" and "toddler"). Once this is done, the environment
should be ready to run the notebooks. In case you want to install the libraries in developer mode
substitute the last line by `pip install -e ./libraries`

To remove the environment and all its installed libraries execute:
```
conda deactivate
conda remove -y --name learning-physical-properties-with-rnns --all
```

# References
Bramley, N. R., Gerstenberg, T., Tenenbaum, J. B., & Gureckis, T. M. (2018). Intuitive experimentation in the physical world. Cognitive Psychology, 105, 9–38. https://doi.org/10.1016/j.cogpsych.2018.05.001


