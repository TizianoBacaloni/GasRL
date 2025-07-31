# GasRL
This is the code to easy replicate the results of [inserisci link], just running the notebooks belows without changes. 
If desired, you can train alternative models and use them to generate figures analogous to those in the paper. 
Below are the steps required to understand and utilize the code:

1) Easy_Train.py streamlines model training by letting you set total steps, auto-save at chosen intervals, and evaluate each checkpoint. Simply tweak the arguments passed to the main() function at the script’s end to adjust steps, checkpoint frequency, learning rates, and more.

2) Easy_Plot.py provides the essential utility functions to seamlessly use and visualize outputs from the other scripts.

3) Gas_Storage_Env.py defines the environment used across the various simulations.

4) Easy_Test.py computes and save data that are need in the following notebooks. Run this script before running them.
   
5) Easy_Price.jpynb compute price volatility.
   
7) Easy_Seasonality.ipynb computes price seasonality using the methods proposed in the paper and compares them against real-world data.


