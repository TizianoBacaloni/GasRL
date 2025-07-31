# GasRL
This code  easily replicates the results of [inserisci link] by running the notebooks below without changes.
The two ZIP files contain the models used to generate the figures.
If desired, you can train alternative models and use them to generate figures analogous to those in the paper. 
Below are the steps required to understand and utilize the code:

1) Train.py streamlines model training by letting you set total steps, auto-save at chosen intervals, and evaluate each checkpoint. Just tweak the arguments passed to the main() function at the script’s end to adjust steps, checkpoint frequency, learning rates, and more.

2) Plots.py provides the essential utility functions to  use and visualize outputs from the other scripts.

3) Gas_Storage_Env.py defines the environment used across the various simulations.

4) Test.py computes and saves data  needed in the following notebooks. Run this script before running them.
   
5) Volatility.jpynb compute price volatility.
   
7) Seasonality.ipynb computes price seasonality using the methods proposed in the paper and compares them against real-world data.

8) Robustness.ipynb lets you compare the robustness of both the unpenalized model (Θₙ = 0) and the penalized model (Θₙ > 0) when the test supply shock (σₛ) exceeds the training value of 0.04.


