# Medical Dead-ends and Learning to Identify High-risk States and Treatments

Machine learning has successfully framed many sequential decision making problems as either supervised prediction, or optimal decision-making policy identification via reinforcement learning. In data-constrained offline settings, both approaches may fail as they assume fully optimal behavior or rely on exploring alternatives that may not exist. Even if optimality is not attainable in such constrained cases, negative outcomes in data can be used to identify behaviors to avoid, thereby guarding against overoptimistic decisions in safety-critical domains that may be significantly biased due to reduced data availability. Along these lines we introduce an approach that identifies possible "dead-ends" of a state space. We focus on the condition of patients in the intensive care unit, where a "medical dead-end" indicates that a patient will expire, regardless of all potential future treatment sequence. We frame the discovery of these dead-ends as an RL problem, training three independent deep neural models for automated state construction, dead-end discovery and confirmation. 

In this repo, we provide the code used develop a novel RL-based method, Dead-end Discovery (DeD), presented in the paper "Medical Dead-ends and Learning to Identify High-risk States and Treatments" published at NeurIPS 2021. DeD focuses on identifying _treatments to avoid_ as opposed to what treatment to select as is typical in RL-based approaches to sequential decision making in healthcare. Our goal in publishing this code is to facilitate reproducibility of our paper in hopes of motivating further research utilizing the DeD framework in safety-critical domains. 

## Paper
If you use this code in your research please cite the following publication [link TBD](https://papers.neurips.cc/):
```
@inproceedings{fatemi2021medical,
 title={Medical Dead-ends and Learning to Identify High-risk States and Treatments},
 author={Fatemi, Mehdi and Killian, Taylor W and Subramanian, Jayakumar and Ghassemi, Marzyeh},
 booktitle={Proceedings of the 35th International Conference on Neural Information Systems},
 year={2021}
}
```

This paper can also be found on arXiv: [TBD](https://arxiv.org)

-----
[LICENSE](https://github.com/microsoft/med-deadend/blob/master/LICENSE)


[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

-----

## Contributing

This code has been developed as part of the RL4H initiative at MSR Montreal. Most of the core work has been done by

- Mehdi Fatemi (mehdi.fatemi@microsoft.com), Senior Researcher, MSR Montreal

with contributions by
- Taylor Killian (twkillian@cs.toronto.edu), Ph.D. Student, University of Toronto

- Jayakumar Subramanian (jayakumar.subramanian@gmail.com), Intern (Summer 2019), MSR Montreal

---

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Requirements

We recommend using the [Anaconda](https://docs.anaconda.com/anaconda/install/) distribution for python dependencies. The code should work will any recent version. In addition, the following packages are required:

- Pytorch (https://pytorch.org/) -> All recent versions should work. Tested on 1.8 and newers. 
- Click (`pip install click`) -> All recent versions should work. Tested on 6.5 and newers.
- Pyprind (`pip install PyPrind`) -> All versions should work. 

## How to use

### Step 0) Data Preparation: 

#### Part 1: MIMIC-III Sepsis Cohort

The code used to define, extract and preprocess the patient cohort from MIMIC-III can be found at [https://github.com/microsoft/mimic_sepsis](https://github.com/microsoft/mimic_sepsis). This code produces two CSV files `sepsis_final_data_K1.csv` and `sepsis_final_data_K1_RAW.csv`.

To replicate the results presented in our paper, it is expected that these two files are accessible via the directory `./data/sepsis_mimiciii/`. 

Then from the root directory, run the following steps (tested on Windows and Linux). All steps depend on the configuration file `config_sepsis.yaml` which is used to define experiment specific parameters as well as the directories where data can be found, artifacts of training and, results are stored. As a default (for illustrative purposes) we name the base experiment as "run1" which is used to create a subfolder.

#### Part 2: Preprocess the extracted trajectories

Run `ipython ./data_process.py`

The patient cohort provided in part 1 is extracted for general purposes. This script places the extracted trajectories into a more convenient format for the DeD learning processes and also constructs independent training, validation and test datasets. 

---

### Step 1) Train the State Construction (SC-) Network

Run `ipython ./train_cortex.py`

Following [Killian, et al.](https://github.com/MLforHealth/rl_representations) we form state representations by processing a sequence of observations, using [AIS](https://github.com/info-structures/ais) prior to and including any time `t` as well as the last selected treatment to form the state `s_t`. The SC-Network is trained to predict the next observation, by reconstructing the latent state representation created via a recurrent encoding module.

-----

### Step 2) Train the Dead-end (D-) and Recovery (R-) Networks

Run `ipython ./train_rl.py -- -f "./results/run1"`

Based on the MDP formulation introduced in the paper, we train two independent DQN models to discover and confirm dead-ends and identify treatments that may leave a patient toward these unfavorable states. The networks are trained using the DDQN approach introduced by [van Hasselt, et al](https://arxiv.org/abs/1509.06461) which can be replaced by any appropriate off-policy or offline RL method.

------

### Step 3) Aggregate and Analyze the Results

#### Part 1) Results aggregation

Run `ipython ./stats/make_plot_data.py`

This script takes the trained D- and R-Networks with the embedded states of the test set to aggregate results and identify potentially high-risk states and treatments. In a generated `./plots` folder two results artifacts are saved:
 - `flag_data.pkl` contains information about states along a patient trajectory that are flagged as medium or high-risk.
 -  `data_value.pkl` contains information about the values of the D- and R- Networks along the step-by-step patient trajectories.

#### Part 2) Generate Figures

 - Run `ipython ./stats/circular_hist.py` to produce the circular histogram in Figure 2.
 - Run `ipython ./stats/base_analysis.py` to aggregate trends of the features and D- + R-Networks, as done in Figure 3.
 - Run `ipython ./stats/make_hist.py` to create the Appendix Figs. A5 and A6 + a raw version of Figure 2.
 - Run `ipython ./stats/tsne.py` to reproduce the T-SNE plots in the Appendix and also generates a list of informative negative trajectories saved to `./plots/good_neg_trajs.yaml`

All following scripts depend on the list of informative negative trajectories: `./plots/good_neg_trajs.yaml`:
 - Run `ipython ./stats/colour_maps.py` to reproduce the colour-maps of Figure 4 and any other desired trajectory from `good_neg_trajs`.
 - Run `ipython ./stats/patients_obs.py` to reproduce the top panel of Figure 4 and any other desired trajectory from `good_neg_trajs`.
 - Run `ipython ./stats/prediction.py` to reproduce the tables found in the appendix (produced in LaTeX format)
