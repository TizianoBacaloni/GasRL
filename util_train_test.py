import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO,SAC,A2C,DDPG,TD3,DQN
from penalty_threshold_gas_storage_env import GasStorageEnv  # Import of the environment defined in the stable_baselines_version.py file
from utils_plot import plot_checkpoint_time_series, plot_aggregate_results 


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
        
        print(f"Modello salvato: {model_snapshot_path}")

        i += 1
        
        if current_total_steps >= total_steps:
           print( f' MODEL FOLDER è {model_folder}')
           print("Raggiunto il numero totale di step. Training terminato.")
           break

    # Test for each checkpoint once training is over
    test_checkpoints(model_folder, n_reps, max_test_steps, model_type)
    print('arrivati a 112')

def test_checkpoints(model_folder, n_reps, max_test_steps, model_type):
    """
    Cerca i checkpoint nella cartella e, per ciascuno,
    esegue i test per n episodi (ognuno lungo max_test_steps).
    Per ogni checkpoint vengono salvate le statistiche sia a livello di serie temporale
    (per il test) sia come valore finale (ultimo step).
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
        print(f" riga 146 Checkpoint path: {checkpoint_path}")
        print(f"\n--- Testing checkpoint: {checkpoint_path} ---")
        
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
        print(f'arrivati a 161 con test_results: ')
        aggregate_results[checkpoint] = test_results["stats"]
        time_series[checkpoint]=test_results["time_series"]

        checkpoint_steps = checkpoint.split('_')[2].split('.')[0]
        print(f" arrivati a 165 Checkpoint steps: ")
        test_folder = os.path.join(model_folder, f"test_{checkpoint_steps}")
        os.makedirs(test_folder, exist_ok=True)
        # Per ogni metrica, plot della serie temporale con banda di CI
        plot_checkpoint_time_series(test_results["stats"], test_folder, checkpoint_steps)


    # Grafici aggregati: usa il valore finale per ciascun checkpoint
    plot_aggregate_results(aggregate_results, model_folder)

    return aggregate_results,time_series  # Return the final stats for the last checkpoint

def run_test(model, n_reps, max_test_steps, ep_metric_keys, sigma = None):
    """
    Definizione della funzione run_test che ora riceve anche sigma come parametro
    """
    # Accumulatori globali per ogni metrica
    agg_mean = {key: [] for key in ep_metric_keys}
    agg_ci   = {key: [] for key in ep_metric_keys}
    agg_std  = {key: [] for key in ep_metric_keys}
    database     = {key: [] for key in ep_metric_keys}

    # Parametri di test
    episodes_data = {key: [] for key in ep_metric_keys}
    
    
    for _ in range(n_reps):
        test_env = GasStorageEnv()
        if sigma is not None:
            old_sigma = test_env.sigma_s
            old_mu = test_env.mu_s
            test_env.sigma_s = sigma  # Imposta il valore corrente di sigma
            test_env.mu_s = old_mu + ((old_sigma**2)-(sigma**2))/2
        obs, info = test_env.reset()
        ep_data = {key: [] for key in ep_metric_keys}
        cumulative_market = 0
        cumulative_reward = 0
        dem_unsati = 0
        wasted_supply = 0
     
        

        for t in range(max_test_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, _, info = test_env.step(action)
            
            for key in ep_metric_keys:
                if key == "reward":
                    cumulative_reward += reward
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
                        ep_data[key].append(info["Inventory"])
                    else:
                        ep_data[key].append(np.nan)
                elif key == "demand unsat":
                    dem_unsati += info["Demand not satisfied"]
                    ep_data[key].append(max_test_steps - dem_unsati)
                elif key == "wasted supply":
                    wasted_supply += info["Supply wasted"]
                    ep_data[key].append(max_test_steps-wasted_supply)
                else:
                    ep_data[key].append(info[key])
            
        test_env.close()
        
        for key in ep_metric_keys:
            episodes_data[key].append(ep_data[key])
        
    print(f"Test completato con {n_reps} ripetizioni")

    for key in ep_metric_keys:
        data = np.array(episodes_data[key])
        mean = np.squeeze(np.nanmedian(data, axis=0))
        std = np.squeeze(np.array(np.nanstd(data, axis = 0)))
        se = std / np.sqrt(n_reps)
        ci = 1.96 * se


        # Aggiorna gli accumulatori globali
        agg_mean[key].append(mean)
        agg_ci[key].append(ci)
        agg_std[key].append(std)

        
    return agg_mean, agg_ci, agg_std, episodes_data

def run_test_general(model, n_reps, max_test_steps):
    """
    Per ogni episodio di test (fino a max_test_steps), si registrano in serie temporale i valori di:
      - reward (ricompensa istantanea, per poi fare il cumulativo)
      - bank_account, inventory, market, demand, supply, excess_demand, demand_shifter, supply_shifter, delta_price
    Se l'episodio termina prima di max_test_steps, il valore finale viene ripetuto fino a raggiungere max_test_steps.
    
    Vengono calcolate due tipologie di statistiche:
      * time_series: per ogni step, la media, std e CI delle metriche calcolata sui vari episodi.
      * final: per ogni metrica, il valore al max_test_steps aggregato (media, std, CI) sui vari episodi.
    """
    # Definisco le metriche che intendo tracciare
    metric_keys = ["reward", "Bank account", "Inventory", "market", "Demand", "Supply", 
               "Excess demand", "Demand Shifter", "Supply Shifter", "Delta Price", "price","reward for the mean", "november inventory","demand unsat","wasted supply"] 
  
    
    agg_mean, agg_ci, agg_std, episodes_data = run_test(model, n_reps, max_test_steps, metric_keys)
    
    
    time_series_stats = {}
    for key in metric_keys:
        time_series_stats[key] = {"mean": agg_mean[key], "std": agg_std[key], "ci": agg_ci[key]}
    
    return {"time_series": episodes_data, "stats": time_series_stats}

    

if __name__ == "__main__":
    
    # test model directory
    model_dir = "test_model_dir_seed"
    
    run_experiment(total_steps=1_000_000, models_dir=model_dir, 
                   n_reps=10, pen=2_00,pen_thresh = 1, model_type="SAC", step_increment = 2_000)

   

