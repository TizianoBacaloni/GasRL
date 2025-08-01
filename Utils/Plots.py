import os
import numpy as np
import matplotlib.pyplot as plt
FIGSIZE = (4, 4)

def plot_sigma_robustness(sigmas, means, cis, metrics):
    for metric in metrics:
        mean_vals = np.array(means[metric])
        ci_vals = np.array(cis[metric])
        axhline = None 

        if metric == 'inventory':
            mean_vals = np.array(means[metric])
            ci_vals = np.array(cis[metric])
            axhline = 2.7
        
        plot_sigma_robustness_single(sigmas, mean_vals, ci_vals, metric, axhline=axhline)


def plot_sigma_robustness_single(sigmas, mean_vals, ci_vals, metric, axhline=None):
    """
    Plots the mean and confidence interval for a given metric across different sigma values.
    
    """

    
    mean_vals = np.array(mean_vals)
    ci_vals = np.array(ci_vals)
    
    plt.figure(figsize=FIGSIZE)
    plt.figure()
    plt.plot(sigmas, mean_vals, 'bo-', label='Mean')
    plt.xlabel("Sigma values")
    plt.ylabel(metric)
    plt.grid(True)
    plt.fill_between(sigmas, mean_vals - ci_vals, mean_vals + ci_vals, color='blue', alpha=0.2, label='95% CI')

    
    # plt.axhline(y=2.4, color='red', linestyle='--', label='Threshold = 2.7')
    if axhline is not None:
        plt.axhline(y=axhline, color='red', linestyle='--', label='threshold')

    plt.legend()
    plt.grid(True)


def plot_time_series(t, mean_vals, ci_vals, metric, checkpoint_steps=None, saving_folder=None, axhline=None):
    
    mean_vals = np.array(mean_vals)
    ci_vals = np.array(ci_vals)
    plt.figure(figsize=FIGSIZE)
    plt.plot(t, mean_vals, 'b-', label='Mean')
    plt.fill_between(t, mean_vals - ci_vals, mean_vals + ci_vals, color='blue', alpha=0.2, label='95% CI')
    plt.xlabel("Test Steps")
    plt.ylabel(metric)
    
    # plt.axhline(y=2.4, color='red', linestyle='--', label='Threshold = 2.7')
    if axhline is not None:
        plt.axhline(y=axhline, color='red', linestyle='--', label='threshold')
        
    

    plt.grid(True)

    # create the folder if it doesn't exist
    if saving_folder is not None:
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        
        plt.savefig(os.path.join(saving_folder, f"{metric}_time_series.png"))
        plt.close()

        print(f"Grafico time series per {metric} salvato in {saving_folder}")

                

def plot_checkpoint_time_series(time_series_stats=None, test_folder=None, checkpoint_steps=None): #Time series is here equal to episode_data
    """
    For each metric in time_series_stats, plot the evolution over time (x-axis = test steps)
    with the mean line and a band given by the confidence interval (CI).
    
    """

    for metric in time_series_stats.keys():
        mean_vals = time_series_stats[metric]["mean"][0][:]
        ci_vals = time_series_stats[metric]["ci"][0][:]
        t = np.arange(len(ci_vals))
        axhline = None 
        if metric == 'inventory':
            axhline = 2.7
        plot_time_series(t, mean_vals, ci_vals, metric, checkpoint_steps, test_folder, axhline=axhline)
        
        
        

def plot_aggregate_results(aggregate_results, model_folder):
    """
    To show the aggregated metrics, we take the last value recorded in the tests performed at each checkpoint.
    I consider the average over the 10 repetitions for each market at step 360
    of the test performed considering the model saved at the 1st checkpoint;
    I consider the average over the 10 repetitions for each market at step 360
    of the test performed considering the model saved at the 2nd checkpoint and so on.
    Then I combine them together as with the dots.
    I obtain:
      1. The trend of the MEAN (with CI band) for each metric.
      2. The trend of the STANDARD DEVIATION for each metric.

    """
    # Keys of aggregate_results are the names of the checkpoints, e.g. "ppo_model_1000000.zip"

    # Extract keys of metrics from the aggregated results
    metric_keys = list(next(iter(aggregate_results.values())).keys())

    # Extract steps from checkpoint names and organize data for each metric
    checkpoints = []
    agg_mean_fin = {key: [] for key in metric_keys}
    agg_ci_fin = {key: [] for key in metric_keys}
    agg_std_fin = {key: [] for key in metric_keys}
    
    for ckpt, stats in sorted(aggregate_results.items(), key=lambda x: int(x[0].split('_')[2].split('.')[0])):
        steps = int(ckpt.split('_')[2].split('.')[0])
        checkpoints.append(steps)
        for key in metric_keys:
            agg_mean_fin[key].append(stats[key]["mean"][0][-1])
            agg_ci_fin[key].append(stats[key]["ci"][0][-1])
            agg_std_fin[key].append(stats[key]["std"][0][-1])
   
    # For each metric, plot the mean with CI and the std
    for key in metric_keys:
        plt.figure(figsize=FIGSIZE)
        means = np.array(agg_mean_fin[key])
        cis = np.array(agg_ci_fin[key])

        plt.plot( checkpoints, means, 'b-o', label='Mean')
        plt.fill_between(checkpoints, means - cis, means + cis, color='blue', alpha=0.2, label='95% CI')
        plt.xlabel("Training Steps")
        plt.ylabel(key)
        plt.title(f"Mean of {key}")

        if key == 'inventory':
            plt.axhline(y=2.4,
                       linestyle='--',
                       linewidth=2,
                       color='red',
                       label='Mimum Threshold = 2.4')


        plt.legend()
        mean_path = os.path.join(model_folder, f"aggregate_{key}_mean.png")
        plt.tight_layout()
        plt.savefig(mean_path)
        plt.close()
        print(f"Mean of {key} saved in {mean_path}")

        # Grafico per la deviazione standard
        plt.figure(figsize=(4, 3))
        stds = np.array(agg_std_fin[key])
        plt.plot(checkpoints, stds, 'r-o', label='Std Dev')
        plt.xlabel("Training Steps")
        plt.ylabel(key)
        plt.title(f"Standard deviation of {key}")
        plt.legend()
        plt.grid(True)
        std_path = os.path.join(model_folder, f"aggregate_{key}_std.png")
        plt.tight_layout()
        plt.savefig(std_path)
        plt.close()
        print(f"Standard deviation of {key} saved in {std_path}")
        
