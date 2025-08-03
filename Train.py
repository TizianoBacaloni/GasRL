import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO,SAC,A2C,DDPG,TD3,DQN
from Utils.Gas_Storage_Env import GasStorageEnv  # Import of the environment defined in the stable_baselines_version.py file
from Utils.Plots import plot_checkpoint_time_series, plot_aggregate_results 




METRIC_KEYS = ["reward", "bank account", "inventory", "market", "demand", "supply", 
               "excess demand", "demand shifter", "supply shifter", "delta price", "price","reward for the mean", "november inventory","demand unsat","wasted supply"] 
  
    


##########################################################################
##########################################################################
## SET MAIN AT THE END OF THE SCRIPT AND RUN IT TO TRAIN-TEST THE MODEL ##
##########################################################################
##########################################################################


def compute_steps_schedule(total_steps: int, step_increment: int) -> list[int]:
    """
        Returns the list of cumulative checkpoints,
        according to the logic: target_steps = min(2**i * step_increment, total_steps).

        :param total_steps: total number of steps to reach
        :param step_increment: base increment used to calculate targets
        :return: list of cumulative steps at each checkpoint
    """

    current_total = 0
    i = 1
    schedule: list[int] = []
    #schedule.append(1)

    while current_total < total_steps:
        target =  (2**i) * step_increment          # cumulative target steps
        if target > total_steps:
            target = total_steps

        
        if schedule and target <= schedule[-1]:     # if the target is already in the schedule, skip it
            break

        schedule.append(target)
        current_total = target
        i += 1

    return schedule

def run_experiment(total_steps, models_dir, n_reps,model_type, pen = None,pen_thresh = None,vol_pen=0, rep_number = 0,   # RUNNNING THE EXPERIMENT
                    step_increment = 5_0):
    
    max_test_steps = 360
    
    # Creation of the main folder for the models
    specs_name = f"model{model_type}_ts{total_steps}_pen{pen}_pen_thresh{pen_thresh}_rep{rep_number}_vol_pen{vol_pen}"
    seed=rep_number+10
    model_folder = os.path.join(models_dir, specs_name)
    
    os.makedirs(model_folder, exist_ok=True)

    # Training environment and model inziialization
    train_env = GasStorageEnv()
    train_env.reset(seed=seed)

    if pen is not None:
        train_env.h = pen
    if vol_pen is not None:
        train_env.theta = vol_pen
    if pen_thresh is not None:
        train_env.penalty_threshold = pen_thresh

    if model_type == "PPO":
        model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./tensorboard/" + specs_name)
    elif model_type == "SAC":
        model = SAC("MlpPolicy", train_env, verbose=1, tensorboard_log="./tensorboard/" + specs_name)
    elif model_type == "A2C":
        model = A2C("MlpPolicy", train_env, verbose=1, tensorboard_log="./tensorboard/" + specs_name)  
    elif model_type == "DDPG":
        model = DDPG("MlpPolicy", train_env, verbose=1, tensorboard_log="./tensorboard/" + specs_name)
    elif model_type == "TD3":   
        model = TD3("MlpPolicy", train_env, verbose=1, tensorboard_log="./tensorboard/" + specs_name)   
    else:
        raise ValueError("model_type must be either 'PPO' or 'SAC'")


    current_total_steps = 0  # Already trained steps
    i = 1                       # Checkpoint index

    # Keep training until the total number of steps is reached
    while current_total_steps < total_steps:
        target_steps =   (2**i) * step_increment
        if target_steps > total_steps:
            target_steps = total_steps

        steps_to_run = target_steps - current_total_steps
        if steps_to_run <= 0:
            break

        print(f"\n=== Training: checkpoint {i} (fino a {target_steps} step, eseguendo {steps_to_run} step) ===")
        model.learn(total_timesteps=steps_to_run, reset_num_timesteps=False,tb_log_name=f"{model_type}_ts{total_steps}_pen{pen}_pen_thresh{pen_thresh}_rep{rep_number}")
        current_total_steps = target_steps

        # Save the model at the current checkpoint
        if model_type=='PPO':
            model_snapshot_path = os.path.join(model_folder, f"ppo_model_{current_total_steps}_pen{pen}_pen_thresh{pen_thresh}_rep{rep_number}.zip")
            model.save(model_snapshot_path)
        elif model_type =="SAC":
            model_snapshot_path = os.path.join(model_folder, f"sac_model_{current_total_steps}_pen{pen}_pen_thresh{pen_thresh}_rep{rep_number}.zip")
            model.save(model_snapshot_path)
        elif model_type == "A2C":
            model_snapshot_path = os.path.join(model_folder, f"a2c_model_{current_total_steps}_pen{pen}_pen_thresh{pen_thresh}_rep{rep_number}.zip")
            model.save(model_snapshot_path)
        elif model_type == "DDPG": 
            model_snapshot_path = os.path.join(model_folder, f"ddpg_model_{current_total_steps}_pen{pen}_pen_thresh{pen_thresh}_rep{rep_number}.zip")
            model.save(model_snapshot_path)
        elif model_type == "TD3":
            model_snapshot_path = os.path.join(model_folder, f"td3_model_{current_total_steps}_pen{pen}_pen_thresh{pen_thresh}_rep{rep_number}.zip")
            model.save(model_snapshot_path)
        
        print(f"Model saved: {model_snapshot_path}")

        i += 1
        
        if current_total_steps >= total_steps:
           print("Reached the total number of steps. Training finished.")
           break

    # Test for each checkpoint once training is over
    test_checkpoints(model_folder, n_reps, max_test_steps, model_type)

def test_checkpoints(model_folder, n_reps, max_test_steps, model_type):
    """
    Find the checkpoints in the model folder and, for each one,
    run tests for n episodes (each up to max_test_steps).
    For each checkpoint, statistics are saved both at the time series level
    (for the test) and as a final value (last step).
    
    """
    # Find all the checkpoint files in the model folder
    if model_type=="PPO":
             checkpoint_files =[f for f in os.listdir(model_folder) if f.startswith("ppo_model_") and f.endswith(".zip")]
             checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    elif model_type =="SAC":
         checkpoint_files =[f for f in os.listdir(model_folder) if f.startswith("sac_model_") and f.endswith(".zip")]
         checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    elif model_type == "A2C":
         checkpoint_files =[f for f in os.listdir(model_folder) if f.startswith("a2c_model_") and f.endswith(".zip")]
         checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    elif model_type == "DDPG": 
         checkpoint_files =[f for f in os.listdir(model_folder) if f.startswith("ddpg_model_") and f.endswith(".zip")]
         checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    elif model_type == "TD3":   
        checkpoint_files =[f for f in os.listdir(model_folder) if f.startswith("td3_model_") and f.endswith(".zip")]
        checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0])) 
    else:
         raise ValueError("the model type you're trying to use is not allowed")

    

    aggregate_results = {}  # Final results for each checkpoint
    time_series={}
    for checkpoint in checkpoint_files:
        checkpoint_path = os.path.join(model_folder, checkpoint)
        
        # Model loading
        if model_type == "PPO":
            model = PPO.load(checkpoint_path)
        elif model_type == "SAC":
            model = SAC.load(checkpoint_path)
        elif model_type == "A2C":
            model = A2C.load(checkpoint_path)
        elif model_type == "DDPG":
            model = DDPG.load(checkpoint_path)
        elif model_type == "TD3":
            model = TD3.load(checkpoint_path)
        
        else:
            raise ValueError("model_type must be either 'PPO' or 'SAC'")
        
        # Run the test for the current checkpoint and save the results
        test_results = run_test_general(model, n_reps, max_test_steps)
        aggregate_results[checkpoint] = test_results["stats"]
        time_series[checkpoint]=test_results["time_series"]

        checkpoint_steps = checkpoint.split('_')[2].split('.')[0]
        test_folder = os.path.join(model_folder, f"test_{checkpoint_steps}")
        os.makedirs(test_folder, exist_ok=True)

        # Plot of the time series for each metric 
        plot_checkpoint_time_series(test_results["stats"], test_folder, checkpoint_steps)


    # Grafici aggregati: usa il valore finale per ciascun checkpoint
    plot_aggregate_results(aggregate_results, model_folder)

    return aggregate_results,time_series  # Return the final stats for the last checkpoint

def run_test(model, n_reps, max_test_steps, ep_metric_keys, sigma = None):

    # Accumulators for each metric
    agg_mean = {key: [] for key in ep_metric_keys}
    agg_ci   = {key: [] for key in ep_metric_keys}
    agg_std  = {key: [] for key in ep_metric_keys}
    database     = {key: [] for key in ep_metric_keys}

    # Ditcionary to collect data for each episode
    episodes_data = {key: [] for key in ep_metric_keys}
    
    
    for _ in range(n_reps):   #Loo over the n_reps repetitions
        test_env = GasStorageEnv()
        if sigma is not None:
            old_sigma = test_env.sigma_s
            old_mu = test_env.mu_s
            test_env.sigma_s = sigma  # Sigma current value
            test_env.mu_s = old_mu + ((old_sigma**2)-(sigma**2))/2
        obs, info = test_env.reset()
        ep_data = {key: [] for key in ep_metric_keys}   #For the current episode (for example, the first of n_reps = 10), I collect the values of each metric over the test’s max_test_steps
        cumulative_market = 0
        cumulative_reward = 0
        dem_unsati = 0
        wasted_supply = 0
     
        

        for t in range(max_test_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, _, info = test_env.step(action)
            
            for key in ep_metric_keys: #Inside the dictionary that, for the current episode, contains the value of the metric for each of the max_test_steps, I choose how to save the value
                if key == "reward":
                    cumulative_reward += reward      # Reward is cumulative, so just take the last of the max_test_steps because it's the cum. sum 
                    ep_data[key].append(cumulative_reward)
                elif key == "reward for the mean":
                    ep_data[key].append((reward))
                elif key == "price":
                    ep_data[key].append(np.exp(action[0]))
                elif key == "market":
                    cumulative_market += info["market"]
                    ep_data[key].append(cumulative_market)
                elif key == 'november inventory':
                    novembers = np.arange(max_test_steps)[9::12]
                    is_november = t in novembers
                    if is_november:
                        ep_data[key].append(info["inventory"])
                    else:
                        ep_data[key].append(np.nan)
                elif key == "demand unsat":
                    dem_unsati += info["demand unsat"]
                    ep_data[key].append(max_test_steps - dem_unsati)
                elif key == "wasted supply":
                    wasted_supply += info["supply wasted"]
                    ep_data[key].append(max_test_steps-wasted_supply)
                else:
                    ep_data[key].append(info[key])
        
            
        test_env.close()
        
        for key in ep_metric_keys:
            episodes_data[key].append(ep_data[key]) # Add the n_reps dictionaries—one for each episode—to the main one: episode_data[key] will have n_reps elements, each of length max_test_steps.
        
    print(f"Test completed with {n_reps} repetitions")

    for key in ep_metric_keys:
        data = np.array(episodes_data[key])           # shape = (n_reps, max_test_steps)
        mean = np.squeeze(np.nanmean(data, axis=0))   # Mean of the max_test_step-lenght-array passo‑per‑passo 
        std = np.squeeze(np.array(np.nanstd(data, axis = 0)))
        se = std / np.sqrt(n_reps)
        ci = 1.96 * se


        # Update the accumulators
        agg_mean[key].append(mean) 
        agg_ci[key].append(ci)
        agg_std[key].append(std)

        
    return agg_mean, agg_ci, agg_std, episodes_data

def run_test_general(model, n_reps, max_test_steps):
    """
    For each episode of test (up to max_test_steps), it records in time series the values of:
      - reward (instantaneous reward, then cumulative)
      - bank_account, inventory, market, demand, supply, excess_demand, demand_shifter, supply_shifter, delta_price
    If the episode ends before max_test_steps, the final value is repeated until max_test_steps is reached.

    Two types of statistics are calculated:
      * time_series: for each step, the mean, std, and CI of the metrics calculated across the various episodes.
      * final: for each metric, the value at max_test_steps aggregated (mean, std, CI) across the various episodes.

    """
    # Metrics to plot
    metric_keys = METRIC_KEYS
    
    agg_mean, agg_ci, agg_std, episodes_data = run_test(model, n_reps, max_test_steps, metric_keys)
    
    
    time_series_stats = {}
    for key in metric_keys:
        time_series_stats[key] = {"mean": agg_mean[key], "std": agg_std[key], "ci": agg_ci[key]}
    
    return {"time_series": episodes_data, "stats": time_series_stats}

    

###########################################################################
############## DEFAULT PARAMETERS YOU CAN CHANGE AS YOU WISH ##############
###########################################################################


if __name__ == "__main__":
    
    # test model directory
    model_dir = "test_model_dir_seed"
    
    run_experiment(total_steps=1_500_000, models_dir=model_dir, 
                   n_reps=5, pen=2_00,pen_thresh = 1_000, model_type="SAC", step_increment = 50)

   

