import torch


class Options(object):
    def __init__(self):

        # ==============================================================
        # DEVICE
        # ==============================================================
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ==============================================================
        # ENVIRONMENT CONFIG  (passed to HydronicHeatingEnv)
        # ==============================================================
        self.env_config = {
            # --- time ---
            'dt': 60,                  # internal timestep (seconds)
            'agent_interval': 15,      # agent acts every 15 internal steps (= 15 min)
            'episode_days': 5,         # default episode length (days)

            # --- action limits ---
            'shift_step_limit': 0.5,   # max |ΔT_shift| per step (K)
            'shift_limit': 5.0,        # max total |T_shift| (K)

            # --- reward weights (built into env) ---
            'alpha': 1.0,              # comfort
            'beta': 0.1,               # energy
            'gamma_r': 0.5,            # shift smoothness
            'delta': 0.1,              # valve oscillation
            'asymmetry': 1.5,          # extra cold penalty
            'hard_band': 2.0,          # hard comfort band ±°C
            'hard_penalty': 5.0,       # penalty outside hard band

            # --- domain randomisation ---
            'randomize': True,
            'UA_env_range': (140.0, 280.0),
            'C_wall_range': (8_000_000.0, 25_000_000.0),
            'UA_radiator_range': (200.0, 400.0),
            'T_mean_range': (0.0, 12.0),
            'A_daily_range': (2.0, 7.0),
            'T_air_init_range': (16.0, 22.0),
            'K_init_range': (1.0, 2.0),
            'T_shift_init_range': (-2.0, 2.0),
            'episode_days_range': (3, 7),
        }

        # ==============================================================
        # SAC HYPERPARAMETERS
        # ==============================================================
        self.total_timesteps = 200_000    # total training steps across all episodes
        self.learning_rate = 3e-4
        self.batch_size = 256
        self.buffer_size = 100_000        # replay buffer
        self.gamma = 0.95                 # discount factor
        self.tau = 0.005                  # soft-update coefficient
        self.ent_coef = "auto"            # auto-tuned entropy
        self.learning_starts = 1_000      # random exploration before learning
        self.train_freq = 1               # update every step
        self.gradient_steps = 1           # gradient steps per update

        # policy network architecture
        self.net_arch = [256, 256]

        # ==============================================================
        # MODEL PATHS
        # ==============================================================
        self.path_to_model_from_root = "trained_models"
        self.model_name = "hydronic_bems_sac"

        # ==============================================================
        # LOGGING (Weights & Biases — optional)
        # ==============================================================
        self.wandb = False
        self.wandb_key = "INSERT_KEY_HERE"
        self.wandb_project = "RL_BEMS_Hydronic"
