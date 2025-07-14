

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

half =torch.tensor(0.5, dtype=torch.float32)
tres = torch.tensor(0.8,dtype=torch.float32)

class GasStorageEnv(gym.Env):
    def __init__(self):

        # Time parameters
        self.N = 12  # Months per year
        self.T_max = self.N * 30  # 30 years

        # Demand parameters
        self.eta_d = 0.20  # Elasticity of demand # TODO: consumatore con maggiore elasticità?
        self.lambda_d = 0.975  # Stickiness of demand
        self.rho_d = 0.98  # Persistence of demand shocks
        self.sigma_d = 0.01  # Volatility of demand shocks

        # Supply parameters
        self.eta_s = 0.30  # Elasticity of supply
        self.lambda_s = 0.95  # Stickiness of supply
        self.rho_s = 0.75  # Persistence of supply shocks # TODO: più alto
        self.sigma_s = 0.04  # Volatility of supply shocks # TODO: più alto del 50%?
        self.mu_s = 0.0
        # TODO: jump diffusion for supply shocks (poisson process additivo, da calibrare alla dimensione delle importazioni)

        ################################
        self.sim_index=0
        
        # Storage parameters
        self.tau = 0.005  # Storage cost per month
        self.I_max = torch.tensor(3.0,dtype=torch.float32) # Storage capacity
        self.r = 1.0025  # Interest rate on cash
        self.theta = 100.0  # Volatility penalty weight
        self.h = 1_000.0  # Market clearing penalty
        self.penalty_thresh = 0 # Threshold penalty
        self.U = 100.0  # Maximum price
        self.L = 1 / 100.0  # Minimum price

        # Working variables
        self.lambda_d_compl = 1 - self.lambda_d
        self.lambda_s_compl = 1 - self.lambda_s

        # Precompute seasonal components
        phi = 2 * np.pi / self.N
        self.seasonal_demand = []
        for t in range(self.T_max):
            seasonality = (0.4276 * np.cos(phi * t) + -0.0122 * np.sin(phi * t) +
                           0.1074 * np.cos(phi * 2 * t) + -0.0003 * np.sin(phi * 2 * t) +
                           -0.0391 * np.cos(phi * 3 * t) + -0.0023 * np.sin(phi * 3 * t) +
                           0.0126 * np.cos(phi * 4 * t) + -0.0347 * np.sin(phi * 4 * t) +
                           0.0302 * np.cos(phi * 6 * t) + 0.0000 * np.sin(phi * 6 * t))
            self.seasonal_demand.append(seasonality)
        self.cos = [np.cos(phi * t) for t in range(self.T_max)]
        self.sin = [np.sin(phi * t) for t in range(self.T_max)]

        # Observation space: [seasonal_demand, cos, sin, u_d, u_s, p_d, p_s, i, old_log_price]
        low = np.array([-1, -1, -1, -10, -10, np.log(self.L), np.log(self.L), torch.log(half), np.log(self.L)])
        high = np.array([1, 1, 1, 10, 10, np.log(self.U), np.log(self.U), torch.log(half+self.I_max), np.log(self.U)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: Price P between L and U
        self.action_space = gym.spaces.Box(low=np.log(self.L), high=np.log(self.U), shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        # Initialize state variables
        self.t = 0
        self.u_d = 0.0  # Demand shifter
        self.u_s = 0.0  # Supply shifter
        self.p_d = 0.0  # Log demand price index
        self.p_s = 0.0  # Log supply price index

        
        self.I = 0.80 * self.I_max # Inventories

        ################################
        self.sim_index = 0
        ####################################
        self.g = 0.0  # Bank account
        self.p_var_cum = 0.0  # Cumulative price variance
        self.penalties_cum = 0.0  # Cumulative penalties
        self.p_prev = 0.0  # Previous log price
        self.demand_shocks = self.sigma_d * np.random.randn(self.T_max) 
        self.supply_shocks = self.sigma_s * np.random.randn(self.T_max) + self.mu_s

        # Initial observation
        state = [self.seasonal_demand[0], self.cos[0], self.sin[0], self.u_d, self.u_s, self.p_d, self.p_s, torch.log(0.5+self.I), self.p_prev]
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        # Action is the log price p
        p = action[0]
        P = np.exp(p)
        delta_p = p - self.p_prev
        p_var = delta_p ** 2
        self.p_prev = p

        old_g = self.g        

        # Update bank account with interest and storage cost
        self.g = old_g *  self.r - self.I * self.tau
        #delta_g = self.g * self.r - self.I * self.tau

        # Update price indices (EWMA)
        self.p_d = np.log(self.lambda_d * np.exp(self.p_d) + self.lambda_d_compl * P)
        self.p_s = np.log(self.lambda_s * np.exp(self.p_s) + self.lambda_s_compl * P)

        # Compute demand and supply
        d = self.u_d + self.seasonal_demand[self.t] - self.eta_d * self.p_d
        s = self.u_s + self.eta_s * self.p_s
        excess_demand = np.exp(d) - np.exp(s)
        spare_storage_capacity = self.I_max - self.I

        # Market clearing check
        demand_is_not_satisfied = (excess_demand > self.I)
        supply_goes_to_waste = (-excess_demand > spare_storage_capacity)
        market_does_not_clear = (demand_is_not_satisfied or supply_goes_to_waste)
        market_clears = (~market_does_not_clear)

        # Penalty for market clearing violation
        penalty = self.h * market_does_not_clear * (1 + supply_goes_to_waste * (-excess_demand - spare_storage_capacity) + demand_is_not_satisfied * (excess_demand - self.I))
        #self.penalties += penalty

        # Update inventory
        delta_I = market_clears * (-excess_demand) + demand_is_not_satisfied * (-self.I) + supply_goes_to_waste * spare_storage_capacity
        self.I = self.I + delta_I

        # Update bank account with transaction costs
        self.g -= P * delta_I
        delta_g = self.g - old_g
        
        # Update shifters
        self.u_d = self.rho_d * self.u_d + self.demand_shocks[self.t]
        self.u_s = self.rho_s * self.u_s + self.supply_shocks[self.t]
        
        # Determine reward and done
        reward = delta_g - self.theta * p_var - penalty
        novembers = np.arange(self.T_max)[9::12]         # [9, 21, 33, ...]
        is_novemeber = self.sim_index in novembers
        inventory_threshold = float(0.83 * self.I_max)
        
        if is_novemeber:
            if self.I < inventory_threshold:
                reward = reward - self.penalty_thresh
                self.penalties_cum += self.penalty_thresh

        self.sim_index = self.sim_index + 1
        
        if self.t < self.T_max - 1:
            done = False
        
        else:
            market_value = 0.5 * (np.exp(self.p_d) + np.exp(self.p_s))
            self.g = self.g + self.I * market_value
            delta_g = self.g - old_g
            reward = delta_g - self.theta * p_var - penalty
            done = True
            

            


        # Next observation
        next_state = [self.seasonal_demand[self.t], self.cos[self.t], self.sin[self.t], self.u_d, self.u_s, self.p_d, self.p_s, torch.log(half+self.I), self.p_prev] 
        self.t += 1
        
        truncated = False

        # add self.I to the info dict
        
        info = {"Inventory": self.I,
                "Reward": reward, 
                "market": market_clears,
                "Bank account":self.g,  
                "Demand":d,
                "Supply":s,
                "Excess demand":excess_demand,
                "Delta Price":p_var,
                "Penalty":penalty,
                "Demand Shifter": self.u_d,
                "Supply Shifter":self.u_s,
                "Supply wasted": supply_goes_to_waste,
                "Demand not satisfied": demand_is_not_satisfied
        }


        
        
        return np.array(next_state, dtype=np.float32), np.array(reward, dtype=np.float32), done, truncated, info

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Function to create environment instances
def make_env():
    return GasStorageEnv()


if __name__ == "__main__":
    
    # Number of parallel environments
    n_envs = 8
    env = DummyVecEnv([make_env for _ in range(n_envs)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True)  # Normalize observations

    # Define PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        clip_range=0.05,
        batch_size = 1024 * 16,
        #gamma=0.99,
        #gae_lambda=0.95,
        n_epochs=1,
        learning_rate=1e-4,
        policy_kwargs={"optimizer_kwargs": {"weight_decay": 1e-5, "betas": (0.99, 0.99)}, "activation_fn": torch.nn.SiLU, "net_arch": dict(pi=[128] * 2, vf=[128] * 2), "log_std_init": -3, "ortho_init": True}
    )

    # Comment this if you want to train the standard deviation of the policy distribution
    #for name, param in model.policy.named_parameters():
    #    if "log_std" in name:
    #        param.requires_grad = False

    # Train the model
    #total_timesteps = int(100_000_000)  
    #model.learn(total_timesteps=total_timesteps)

    test_env = GasStorageEnv()
    def test_model():
        rewards = []
        for n_sample in range(10):
            obs, info = test_env.reset()
            total_reward = 0.0
            for _ in range(360):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)
        print("Test episode total reward: " + "{:,.2f}".format(np.mean(rewards)))

    test_model()
    model.learn(total_timesteps=1024*16)
    for j in range(1000):
        print(f"Training iteration {j}")
        model.learn(total_timesteps=1024*16, reset_num_timesteps=False)
        test_model()

    # Save the trained model
    model.save("ppo_gas_storage_model")
