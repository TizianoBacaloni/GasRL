# GasRL
This code  easily replicates the results of [inserisci link] by running the notebooks below without changes.
The two ZIP files contain the models used to generate the figures:  
**`sac_model_1500000_pen1000.0_pen_thresh1000.0_rep1.zip`** is used just in `Robustness.py`,  
while the baseline specification **`sac_model_1500000_pen1000.0_pen_thresh0.0_rep4.zip`**  
is used in all notebooks and scripts.

If desired, you can train alternative models and use them to generate figures analogous to those in the paper. 
Below are the steps required to understand and utilize the code, with [] specifing the repository of the file:

1) `Train.py`  streamlines model training by letting you set total steps, auto-save at chosen intervals, and evaluate each checkpoint. Just tweak the arguments passed to the main() function at the script’s end to adjust steps, checkpoint frequency, learning rates, and more.

2) `Plots.py` [Utils]provides the essential utility functions to  use and visualize outputs from the other scripts.

3) `Gas_Storage_Env.py` [Utils] defines the environment used across the various simulations.

4) `Test.py` computes and saves data  needed in the following notebooks. Run this script before running them.
   
5) `Volatility.jpynb` [Plots] compute price volatility.
   
7) `Seasonality.jpynb`  [Plots] computes price seasonality using the methods proposed in the paper and compares them against real-world data.

8) `Robustness.py` [Plots] lets you compare the robustness of both the unpenalized model (Θₙ = 0) and the penalized model (Θₙ > 0) when the test supply shock (σₛ) exceeds the training value of 0.04.

9) **`sac_model_1500000_pen1000.0_pen_thresh1000.0_rep1.zip`** and **`sac_model_1500000_pen1000.0_pen_thresh0.0_rep4.zip`**  [Models] are the models used for the simulations of the paper.


