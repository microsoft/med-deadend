# LifeGate
*Tabular example* for the medical dead-end paper

# How to run:

- In the lifegate folder, run `ipython ./train.py` to train the agent. 
- By default the results will be saved in the `results` sub-folder with an increasing-numbered name for each run.
- See `config.yaml` for the global params. These params can also be supplied directly by (possibly multiple) use of flag `-o`. For example:
     - `ipython ./train.py -- -o folder_name myname -o gamma 0.95 -o alpha 0.05`
- Use the `Visualizing Q_D and Q_R.ipynb` notebook for plotting.
- Outputs of the sample run we used in the paper are provided under `results`. 
