import numpy as np
import os
import re
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO,SAC,A2C,DDPG,DQN,TD3,HER
from util_train_test import run_test,run_test_general,compute_steps_schedule
from utils_plot import plot_checkpoint_time_series

TRAINING_STEPS = 1_500_000
STEP_INCREMENT = 2


#FINISICI COMMENTARE BENE RIGA 183


####################################################
## CAMBIA RIGA 60 SE NON STAI IN EXPERIMENTS ##
####################################################

def load_model(dir, model_name): 
    
    """The function loads the appropriate algorithm type based on the suffix of the model names"""

    if "PPO" in dir:
        return PPO.load(model_name)
    elif "SAC" in dir:
        return SAC.load(model_name)
    elif "A2C" in dir:
        return A2C.load(model_name)
    elif "DDPG" in dir:
        return DDPG.load(model_name)
    elif "TD3" in dir:
        return TD3.load(model_name)
    else:
        raise ValueError(f"Tipo di modello non riconosciuto nel filename: {dir}")

def get_checkpoint_paths(experiments_dir):
    
    """
    For the various trained and saved models (e.g., zip files), 
    the function allows the identification and saving of both 
    the paths and the names of the directories where they are located.
    
    """
    
    check_paths = []
    if not os.path.isdir(experiments_dir):
        raise FileNotFoundError(f"Directory non trovata: {experiments_dir}")

    for sub in os.listdir(experiments_dir):          # For every sub-directory in experiments_dir (es. modelA2C_ts10000000_pen1000.0_rep0)
        sub_dir = os.path.join(experiments_dir, sub) # Path of the sub-directory=join between experiments_dir path and sub-direcotry name

        for fname in os.listdir(sub_dir):            # For each file in the current sub-direcotry
            if fname.endswith('.zip') :              
                check_paths.append(os.path.join(sub_dir, fname))  # Append the path of the file to check_paths list
                
                
    dirs  = [p.split("experiments20/",1)[1].split("/",1)[0] for p in check_paths]    # Direcotory name =  first part of  relative check_path once splitted in two parts after "experiments1"
 
    return check_paths, dirs

def get_final_checkpoint_paths(experiments_dir):

    """
    1) The function selects and return mostr-trained models' paths and dirs
    
    2) If {final_steps}-trained model is not in a certain directory (the training
    is not completed) the most trained alternative is selected.
    """


    final_steps = str(1000000)
    alternatives: list[str] = [4_096_000, 8_192_000]
    
    check_paths,dirs = get_checkpoint_paths(experiments_dir=model_dir)       
    unique_dirs = list(dict.fromkeys(dirs))     #ensure  directories are unique removing duplicates to avoid modelSAC_ts10000000_pen3000.0_rep1 for both modelSAC_ts10000000_pen3000.0_rep2/ppo_10.zip and  modelSAC_ts10000000_pen3000.0_rep1/ppo_20.zip

    trainest_path = []
    final_models = {}
    for d in unique_dirs:
        # 1) Filtering just paths in which d is present
        cand = [p for p in check_paths if d in p]  #Candidates paths
        
        # 2) final_step value is in every name if the model is trained on final_steps value (i.e modelA2C_ts{final_steps}_pen1000.0_rep2/ppo_model_2048000.zip)
        matches_ten = [p for p in cand if (final_steps) in os.path.basename(p)] # Paths containing final_steps value in the basename
        matches_alt = [p for p in cand if any(str(alt) in (p) for alt in alternatives)] #Paths containing alternatives values in their entire name

        final_models[d] = {"final model": matches_ten, "alternative": matches_alt}

        if final_models[d]["final model"]:
            trainest_path.append(final_models[d]["final model"])
            print('Getting 10 milions model')
        
        elif final_models[d]["alternative"][1]: # If 10mln trained model not found, 1st alternative
            trainest_path.append(final_models[d]["alternative"][1])
            print('Getting 8 milions model')
        
        else:
            trainest_path.append(final_models[d]["alternative"][0]) #If 10mln trained model not found, 2nd alternative
            print('Getting 4 milions model')
        

    return trainest_path,unique_dirs,final_models

def get_metrics_values(mean,std,data,model_keys,metric_keys,max_test_steps,thresh,saving_dict=None):

    """
    1) The function computes, for every metric, the final value of the test-time-series and saves it inside saving_dict
    with the (model_keys, metric) keys pair

    2) For "Inventory" the saved value is given by the mean computed on every 10th month (i.e the first day of the 11th)
    for every years

    3) Considering {  "Delta Price" = (log(P)-log(Pt-1))^2  } the volatility measure is computed as the mean of sample std
    across the n_reps of the mean-values along the max_test_steps (i.e the std across n_reps values each equal to the mean on max_test_steps)

##### SERIE STORICA DELLA DIFFERENZA DEI LOG PREZZI
    """

    novembers = np.arange(max_test_steps)[9::12]      # The first value is the 10th element (1st november) and than one value every 12 (i.e  1st november of the next year)
    for key in metric_keys:
            
            #mean[key] and std[key] are bi-dimensional array with the first-one equal to the real arrays of values ----> just first dimension to consider ---->  mean[key][0]
            
            if key == 'Inventory':       # For Invenotries mean and std are computed considering, for the n_reps, the amount of gas stored at the end every november
                saving_dict[model_keys][key] = {"mean": np.mean(mean[key][0][novembers]), 
                                                    "std": np.mean(std[key][0][novembers]),
                                                    "under threshold":np.sum(mean[key][0][novembers] < thresh)} # How often it fails to reach the threshold
            elif key == "Delta Price":
                delta_prices = data["Delta Price"] #  data is a dictionary with one key for each metric: inside each key the data-shape  is (n_reps, max_test_steps)
                delta_prices = np.array(delta_prices) 
                sum_delta_prices = np.sum(delta_prices, axis=1) # For each of n_reps, the sum over max_test_steps (row sum): the result are n_reps single values
                
                var_delta_prices = sum_delta_prices / delta_prices.shape[1]   # Sample variance
                std_delta_prices = np.sqrt(var_delta_prices)
                
                mean_std_delta_prices = np.mean(std_delta_prices)             # Final volatility measure
                std_std_delta_prices = np.std(std_delta_prices)
                
                saving_dict[model_keys][key] = {"mean": mean_std_delta_prices, 
                                                    "std": std_std_delta_prices,
                                                        "serie originale": std_delta_prices}  
            else:
                saving_dict[model_keys][key] = {"mean": (np.array(mean[key][0][-1])), "std": np.squeeze(np.array(std[key][0][-1]))}

    return saving_dict
    
def get_data_as_dictionary(models, dir_names,metric_keys):

    """
    The function iterates run_test and get_metrics values across each model in models

    """
    data = {}
    
    for dir_name, model_name in zip(dir_names, models):
        data[model_name] = {}        # One dictionary for each model, with mean and std inside
        fname = os.path.basename(model_name)    # Name of the zip file which is used as model
        
        #print(f'il nome del modello che sto girando è {dir_name} con modello {fname}') 

        model = load_model(dir_name, model_name)  # Load the model from the checkpoint
        
        agg_mean, agg_ci, agg_std, episodes_data = run_test(model, n_reps, max_test_steps, metric_keys) # Considerando la reward in ciascuno step di test come la reward cumulativa, prendo solo l'ultimo valore, che è la sommatoria di tutti gli altri
        
        #print(f"\033[31mCI SIAMO QUASI SPERO ED HO PER {model_name} un bel {agg_mean} E COME SCORDARE {episodes_data}\033[0m")

        data = get_metrics_values(agg_mean,agg_std,episodes_data,model_name,metric_keys,max_test_steps,thresh,saving_dict=data) # Creo un dizionario tipo questo {
                                                                                                                                                                    #modelPPO_ts5000000_pen2000.0_pen_thresh0.0_rep1/ppo_model_1638400_pen2000.0_pen_thresh0.0_rep1.zip': 
                                                                                                                                                                                                                                                                        #{'reward': {'mean': array(-178.99048, dtype=float32), 'std': array(0., dtype=float32)}, 
                                                                                                                                                                                                                                                                        # 'Inventory': {'mean': np.float32(nan), 'std': np.float32(nan), 'under threshold': np.int64(0)}, 
                                                                                                                                                                                                                                                                        # 'market': {'mean': array(3.), 'std': array(0.)}, 
                                                                                                                                                                                                                                                                        # 'Delta Price': {'mean': np.float32(0.7733184), 'std': np.float32(0.0)}}, 
                                                                                                                                                                    
                                                                                                                                                                    #/modelPPO_ts5000000_pen2000.0_pen_thresh0.0_rep1/ppo_model_204800_pen2000.0_pen_thresh0.0_rep1.zip': 
                                                                                                                                                                                                                                                                      # {'reward': {'mean': array(-83.372185, dtype=float32), 'std': array(0., dtype=float32)}, 
                                                                                                                                                                                                                                                                   #'Inventory': {'mean': np.float32(nan), 'std': np.float32(nan), 'under threshold': np.int64(0)}, 
                                                                                                                                                                                                                                                                     #  'market': {'mean': array(3.), 'std': array(0.)}, 
                                                                                                                                                                                                                                                                  # 'Delta Price': {'mean': np.float32(0.5288841), 'std': np.float32(0.0)}}, 
        
        save_root = "data_dictionary"
        os.makedirs(save_root, exist_ok=True)
        np.save(os.path.join(save_root, "data_dictionary.npy"), data)  # Save the alg_step_values dictionary to a .npy file
        print(f"[OK] Salvato database")                                                                                                                                                         #}
    return data

def best_models_criteria(database):

    """
    1) The function selects the model maximizing/minimizing a given metric
    and append its path to the corrispondent list.

    2) It also append to why_selected_models the reason why the model is selected

    """
    
    best_model_paths = []  # List to store the identifiers of selected best models
    why_selected_models = []  # List to record the reason each model was selected

    # Find the model with the lowest volatility (based on the mean Delta Price)
    lowest_volatility_model = min(database, key=lambda k: database[k]["Delta Price"]["mean"])
    why_selected_models.append("Lowest volatility")  # Explain why this model was selected
    best_model_paths.append(lowest_volatility_model)  # Store the selected model

    # Print the result with its corresponding metric value
    print(f'Model with the lowest volatility: {lowest_volatility_model} (std = {database[lowest_volatility_model]["Delta Price"]["mean"]})')

    # Find the model with the highest reward (based on mean reward value)
    best_reward_model = max(database, key=lambda k: database[k]["reward"]["mean"])
    why_selected_models.append("Best reward")  # Explain the selection reason
    best_model_paths.append(best_reward_model)  # Store the selected model
    print(f"Model with the highest reward: {best_reward_model} (mean = {database[best_reward_model]['reward']['mean']})")

    # Find the model with the highest market metric
    best_market_model = max(database, key=lambda k: database[k]["market"]["mean"])
    why_selected_models.append("Best market")  # Explain the selection reason
    best_model_paths.append(best_market_model)  # Store the selected model
    print(f"Model with the highest market value: {best_market_model} (mean = {database[best_market_model]['market']['mean']})")

    # Return all selected model names, the database, and selection reasons
    return database, lowest_volatility_model, best_reward_model, best_market_model, best_model_paths, why_selected_models

def best_models_selection( metric_keys): 
    
   
    rep_number_data= get_data_as_dictionary(paths,dirs,metric_keys)
    rep_number_data, low_vol_mod, best_rew_mod, best_market_mod, bst_mod_paths,why_selected_mods = best_models_criteria(rep_number_data)
    
    
    return rep_number_data, low_vol_mod, best_rew_mod, best_market_mod, bst_mod_paths,why_selected_mods

def plot_best_model(best_models,characteristics, paths, n_reps, max_test_steps):

    """
    dirs = the entire file  path of ALL models (es. gas_storage_model/experiments1/modelSAC_ts10000000_pen1000.0_rep0/ppo_model_10000000.zip )
    best_models = the entire file path of just BEST models
    charachteristics = the reason why the model is seleceted as one of the best
    """
    
     
    for path, model_name,choice_reason in zip(paths, best_models,characteristics):  
        
        model = load_model(path, model_name) 
        
        tipo_modello = choice_reason
        general_model_name = path.split("/")[-2] #es. SAC_ts10000000_pen3000.0_rep0
        model_file = path.split("/")[-1]         #es. sac_model_81920000.zip
        general_model_name = general_model_name.replace(".zip", "")
    
        test_result=run_test_general(model, n_reps, max_test_steps)
        plot_checkpoint_time_series(test_result["stats"], test_folder=f"{tipo_modello}_{general_model_name}_{model_file}", checkpoint_steps=None) #Plotting metric test-mean evolution with corrispondent CI

def get_data_for_best_algorithm( paths,algorithms):

    """
    The function returns a dictionary for each alghoritm
    with two keys for each metric (i.e mean and std)
    """
    
    rep_number_data = get_data_as_dictionary(paths, dirs, metric_keys) #Working data
     
    alg_stats = {}
    #1) Inizialization of one sub-dictionary for each algorithm
    for alg in algorithms:
        alg_stats[alg] = {}
    #2) Storage of different metric_mean and metric_std for each algorithm in its sub-dictionary
        for metric in metric_keys:
            alg_stats[alg][f"{metric}_means"] = []
            alg_stats[alg][f"{metric}_stds"] = []

    for key, data_block in rep_number_data.items():     # Iterate over each entry in rep_number_data, where `key` identifies the model/run and `data_block` holds its metrics
        for alg in algorithms:                          # Check each algorithm name of interest in the list `algorithms`

            if alg in key:                              # Accessing the first-level key: each alg in algorithm is a key
                for metric in metric_keys:              # Iterate through each metric (e.g., 'reward', 'market', etc.) to extract stats
                    
                    mean_val = data_block.get(metric, {}).get("mean")  # Safely retrieve the 'mean' value for this metric; returns None if missing
                    std_val = data_block.get(metric, {}).get("std")    # Safely retrieve the 'std' value for this metric; returns None if missing
                    
                    alg_stats[alg][f"{metric}_means"].append(mean_val) # Append the mean value to the corresponding list in alg_stats for this algorithm
                    

                    alg_stats[alg][f"{metric}_stds"].append(std_val)   # Append the std value to the corresponding list in alg_stats for this algorithm
    return alg_stats

def best_alghoritms(algorithms, alg_statistics):
    """
    1) The function considers a specific metric (e.g., the reward mean), the function computes 
    the test-mean value of that metric across all models of the same algorithm type 
    (e.g., PPO), irrespective of the number of training steps.
    
    2) Than, it selects the alghoritm able to provide the best value for the considered metric. 
      """
    
    aggregate_statistics = {}           # General dictionary, same for each alg.

    for alg in algorithms:
        for metric in alg_statistics[alg]:
            aggregate_statistics.setdefault(metric, {})[alg] = np.mean(alg_statistics[alg][metric])   # If aggregate_statistics doesn't contain the metric yet, creation of a correspondent dictionary  inside aggregate_statistics
                                                                                                      # Inside the metric dictionary a new key is created, corresponding to the alg type
                                                                                                      # The value associated with the key is the mean, for the considered metric, of it's value across every model of the same alghoritm


    best_per_metric = {                                                                               # Build a dict mapping each metric to the algorithm with the highest mean score
    metric: max(vals.items(), key=lambda x: x[1])                                                     # For each metric and its corresponding dict of {algorithm: mean_value}…
                                                                                                      # Find the (algorithm, mean_value) pair with the maximum value component
    for metric, vals in aggregate_statistics.items()
}
    
    print(f'{best_per_metric}')


        
def best_seed (moels_dir,metric_keys):

    """
    A general dictiionary seed_dict is created, in which there is one 1st order key for each subfolder inside models_dir: than, there is one 
    2nd level key for each metric.
    The value of the 2nd level key is a dictionary with the mean and std of the metric.
    The function compute, for each 1st order key, the mean and std of the metric across all models inside the subfolder.
    Finally, it returns the 1st order key with the best mean value for each metric.
    """


    seed_dict = {}  # Initialize an empty dictionary to store the results

    for subfolder in os.listdir(moels_dir):  # Iterate through each subfolder in the models directory
        subfolder_path = os.path.join(moels_dir, subfolder)  # Get the full path of the subfolder
        if not os.path.isdir(subfolder_path):  # Check if the path is a directory
            continue  # Skip if it's not a directory
        
        seed_dict[subfolder] = {}  # Initialize a dictionary for this subfolder
        
        for model_file in os.listdir(subfolder_path):  # Iterate through each model file in the subfolder
            if not model_file.endswith('.zip'):  # Check if the file is a zip file
                continue  # Skip if it's not a zip file
            
            model_path = os.path.join(subfolder_path, model_file)  # Get the full path of the model file
            print(model_path)  # Print the model path for debugging
            model = load_model(model_path,model_path)  # Load the model
            
            agg_mean, agg_ci, agg_std, episodes_data = run_test(model, n_reps, max_test_steps, metric_keys)  # Run the test
            
            # Initialize a sub-dictionary for each metric
            for metric in metric_keys:
                if metric not in seed_dict[subfolder]:
                    seed_dict[subfolder][metric] = {"mean": [], "std": []}
                # Append the mean and std values for the metric
                seed_dict[subfolder][metric]["mean"].append(agg_mean[metric][0][-1])  # Append the last value of the metric mean
                seed_dict[subfolder][metric]["std"].append(agg_std[metric][0][-1])  # Append the last value of the metric std
    # Now compute the mean and std for each metric across all models in each subfolder
    best_seed_dict = {}  # Initialize a dictionary to store the best seed for each metric
    for subfolder, metrics in seed_dict.items():
 

        best_seed_dict[subfolder] = {}
        for metric, values in metrics.items():
            mean_value = np.mean(values["mean"])
            std_value = np.std(values["std"])
            best_seed_dict[subfolder][metric] = {"mean": mean_value, "std": std_value}  # Store the mean and std for each metric in the subfolder
        
    # Now find the best seed for each metric
    best_seed = {}
    for metric in metric_keys:
        best_seed[metric] = max(best_seed_dict.items(), key=lambda x: x[1][metric]["mean"])
    return best_seed  # Return the dictionary containing the best seed for each metric
                         

def compute_reward_vs_steps( rep_number_data,algorithms, metric_keys=['reward']):

    steps = compute_steps_schedule(TRAINING_STEPS,STEP_INCREMENT)   
    #steps = [4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]                                                                # Generate list of cumulative training steps
    
    # Initialize a struct*9ure to collect reward means for each algorithm at each step
    alg_step_values = {alg: {step: [] for step in steps} for alg in algorithms}                                                     # Creating a dictionary with alg as 1st level key and step as 2nd level key
                                                                                                                                    #    'PPO':   {2000: [], 4000: [], …},
                                                                                                                                    #    'SAC':   {2000: [-35,-54], 4000: [-345,-54], …},
                                                                                                                                    #    …
                                                                                                                                    #    }
    
    # Populate alg_step_values
    for key, data_block in rep_number_data.items():                                                                                 # Iterate over two items of the dict: the list of algorithms and the list of steps
        for alg in algorithms:
            if alg in key:                                                                                                          # If alg is in key, where key is modelSAC_ts5000000_pen2000.0_pen_thresh0.0_rep0/sac_model_25600_pen2000.0_pen_thresh0.0_rep0.zip
                for step in steps:                                                                                                  # The previous line allows to define what 1st level key the code must refer to
                    if f"{alg}_model_{step}_" in key:                                                                               # GIven a 1st order key, for all models of the same algorithm, check if the 2nd level key is in the key (just consider what comes after  the / in the key)
                        for metric in metric_keys:
                        # Get mean reward if available
                            print(f' per il modello {key} ho {rep_number_data[key][metric]['mean']}')
                            mean_val = data_block.get(metric, {}).get("mean")                                                       # Once defined 1st and 2nd level keys, get the  3rd level key, so the mean value of the metric, 
                            if mean_val is not None:
                                alg_step_values[alg][step].append(mean_val)                                                         # Appendin the value inside like  {
                                                                                                                                    #    'PPO':   {2000: [], 4000: [], …},
                                                                                                                                    #    'SAC':   {2000: [-35,-54], 4000: [-345,-54], …},
                                                                                        #    }
    
    # CREATE A FOLDER TO SAVE THE DATABASE ALG_STEP_VALUES
    save_root = "alg_step_values-ALL-ALG"
    os.makedirs(save_root, exist_ok=True)
    np.save(os.path.join(save_root, "alg_step_values.npy"), alg_step_values)  # Save the alg_step_values dictionary to a .npy file
    print(f"[OK] Salvato database dei valori di reward per step: {os.path.join(save_root, 'alg_step_values.npy')}")



    

if __name__ == "__main__":
    #model_dir = "/home/tizianobacaloni/tiziano_data/RL4GasStorage/gas_storage_model/experiments3"  
    model_dir = "/home/tizianobacaloni/tiziano_data/RL4GasStorage/gas_storage_model/experiments20"  

    best_seed_dir = "/home/tizianobacaloni/tiziano_data/RL4GasStorage/gas_storage_model/experiments25 copy"

    paths,dirs = get_checkpoint_paths(experiments_dir=model_dir)
    #trainest_path,final_dirs,final_checkpoints = get_final_checkpoint_paths(experiments_dir=model_dir) 
    
    metric_keys = ["reward","Inventory", "market","Delta Price"] 
    n_reps = 5
    max_test_steps = 360
    thresh = 2.4
    
    # Esegui il test sui modelli
    #print("Esecuzione del test sui checkpoint finali...")
    #results, low_volatility, bst_reward, bst_market,best_paths,why_selected = best_models_selection( metric_keys)
    #results2 = get_data_as_dictionary(paths, dirs, metric_keys) # Get the data for all models
    

    #best_models = [low_volatility, bst_reward, bst_market]
    #plot_best_model(best_models,why_selected, best_paths, n_reps, max_test_steps)

    algorithms = ["ppo", "sac", "a2c", "ddpg", "td3"]
    #alg_stats = get_data_for_best_algorithm(paths,algorithms)

    best_seed = best_seed(best_seed_dir, metric_keys)  # Get the best seed for each algorithm
    print(f"Best seed for each algorithm: {best_seed}")
    
    
    #best_alghoritms(algorithms,alg_stats)

    #compute_reward_vs_steps(results2,algorithms, metric_keys=['reward'])       # Prendo solo l'ultimo valore ma la reward è cumulativa
    #print(results2.item())



  # Stampa i risultati


#    print("Risultati del test:")
#    for model, metrics in results2.items():
#        print(f"Modello: {model}")
#        for metric, values in metrics.items():
#            if metric == "Delta Price":
#                print(
#                    "\033[31m"  
#                    + f"  Price volatility  Media: {values['mean']}, Deviazione Standard: {values['std']}"
#                    + "\033[0m"  
#                )
#            elif metric == 'Inventory':
#                print(
#                    "\033[1;33m"  # start bold + yellow
#                    + f"Metrica: {metric}, Media: {values['mean']}, Deviazione Standard: {values['std']}, Novembers under threshold: {values['under threshold']}"
#                    + "\033[0m"   
#                )
#            else:
#                print(
#                    "\033[1;32m"  # start bold green
#                    + f"  Metrica: {metric}, Media: {values['mean']}, Deviazione Standard: {values['std']}"
#                    + "\033[0m"   # reset to default
#)

#print("\033[1;32m"
#      + f'Nella result 2 io ho un delta price che è {results2["/home/tizianobacaloni/tiziano_data/experiments6/modelSAC_ts5000000_pen2000.0_pen_thresh0.0_rep1/sac_model_1024000_pen2000.0_pen_thresh0.0_rep1.zip"]["Delta Price"]} '
#      + "\033[0m")           
