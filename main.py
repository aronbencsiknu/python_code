# %% imports
import pathlib
import argparse
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from environment import HydronicHeatingEnv
from options import Options
from plot import Plot


# %%  ==================  BASELINE CONTROLLER  =========================

class BaselineController:
    """
    Non-RL baseline: fixed heating-curve slope with a slow PI
    on room-temperature error that nudges the curve shift.
    """

    def __init__(self, Kp_room=0.1, Ki_room=0.001, base_shift=0.0,
                 shift_step_limit=0.5):
        self.Kp_room = Kp_room
        self.Ki_room = Ki_room
        self.base_shift = base_shift
        self.shift_step_limit = shift_step_limit
        self.integral = 0.0
        self.current_shift = base_shift

    def predict(self, T_in, T_set):
        """Return a continuous ΔT_shift action as np array."""
        error = T_set - T_in
        self.integral += error
        self.integral = float(np.clip(self.integral, -50.0, 50.0))

        desired_shift = self.base_shift + self.Kp_room * error + self.Ki_room * self.integral
        desired_delta = desired_shift - self.current_shift
        desired_delta = float(np.clip(
            desired_delta, -self.shift_step_limit, self.shift_step_limit
        ))
        self.current_shift += desired_delta
        return np.array([desired_delta], dtype=np.float32)

    def reset(self):
        self.integral = 0.0
        self.current_shift = self.base_shift


# %%  ==================  OPTIONAL WANDB CALLBACK  =====================

class WandbRewardCallback(BaseCallback):
    """Logs episode rewards to W&B if enabled."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                try:
                    import wandb
                    wandb.log({
                        "episode_reward": info["episode"]["r"],
                        "episode_length": info["episode"]["l"],
                    })
                except Exception:
                    pass
        return True


# %%  ==================  PARSE ARGUMENTS  =============================

parser = argparse.ArgumentParser(description="RL-BEMS: Hydronic Heating with SAC")
parser.add_argument("-p", "--pretrain",
                    help="train a new SAC model from scratch",
                    action="store_true")
parser.add_argument("-l", "--localdemo",
                    help="run a local demo with a trained model",
                    action="store_true")
parser.add_argument("-b", "--baseline",
                    help="run the baseline controller for comparison",
                    action="store_true")
parser.add_argument("-wb", "--wandb",
                    help="log training to Weights & Biases",
                    action="store_true")
args = parser.parse_args()


# %%  ==================  INITIALISE  ==================================

opt = Options()
plotting = Plot()

# resolve paths
current_dir = pathlib.Path(__file__).resolve().parent
project_dir = current_dir.parent
model_dir = project_dir / opt.path_to_model_from_root
model_path = model_dir / opt.model_name


# %%  ==================  TRAINING  ====================================

if args.pretrain:

    print("=" * 60)
    print("  SAC Training — Hydronic Heating RL-BEMS")
    print("=" * 60)
    print(f"  Device        : {opt.device}")
    print(f"  Timesteps     : {opt.total_timesteps:,}")
    print(f"  Batch size    : {opt.batch_size}")
    print(f"  Buffer size   : {opt.buffer_size:,}")
    print(f"  Net arch      : {opt.net_arch}")
    print(f"  Learning rate : {opt.learning_rate}")
    print(f"  Gamma         : {opt.gamma}")
    print("=" * 60, "\n")

    # create environment
    env = HydronicHeatingEnv(opt.env_config)

    # optional W&B
    callbacks = []
    if args.wandb:
        import wandb
        wandb.login(key=opt.wandb_key)
        wandb.init(project=opt.wandb_project, group="SAC_training",
                   settings=wandb.Settings(start_method="thread"))
        callbacks.append(WandbRewardCallback())

    # create SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=opt.learning_rate,
        batch_size=opt.batch_size,
        buffer_size=opt.buffer_size,
        gamma=opt.gamma,
        tau=opt.tau,
        ent_coef=opt.ent_coef,
        learning_starts=opt.learning_starts,
        train_freq=opt.train_freq,
        gradient_steps=opt.gradient_steps,
        policy_kwargs=dict(net_arch=opt.net_arch),
        device=opt.device,
        verbose=1,
    )

    # train
    model.learn(
        total_timesteps=opt.total_timesteps,
        callback=callbacks if callbacks else None,
        progress_bar=True,
    )

    # save
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    if args.wandb:
        wandb.finish()

    # plot last episode
    plotting.plot(env)


# %%  ==================  LOCAL DEMO  ==================================

elif args.localdemo:

    print("=" * 60)
    print("  SAC Demo — Hydronic Heating RL-BEMS")
    print("=" * 60)

    # load trained model
    print(f"  Loading model: {model_path}")
    model = SAC.load(str(model_path), device=opt.device)

    # create fresh environment
    env = HydronicHeatingEnv(opt.env_config)
    obs, info = env.reset()

    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

    avg_reward = total_reward / max(step_count, 1)
    print(f"\n  Steps         : {step_count}")
    print(f"  Total reward  : {total_reward:.2f}")
    print(f"  Avg reward    : {avg_reward:.4f}")
    print(f"  Energy used   : {env.energy_cumulative:.1f} kWh\n")

    plotting.plot(env)


# %%  ==================  BASELINE COMPARISON  =========================

elif args.baseline:

    print("=" * 60)
    print("  Baseline Controller — Hydronic Heating RL-BEMS")
    print("=" * 60)

    env = HydronicHeatingEnv(opt.env_config)
    obs, info = env.reset()

    baseline = BaselineController(
        shift_step_limit=opt.env_config['shift_step_limit']
    )
    baseline.reset()

    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        action = baseline.predict(env.building.T_air, env.heating_curve.T_room_set)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

    avg_reward = total_reward / max(step_count, 1)
    print(f"\n  Steps         : {step_count}")
    print(f"  Total reward  : {total_reward:.2f}")
    print(f"  Avg reward    : {avg_reward:.4f}")
    print(f"  Energy used   : {env.energy_cumulative:.1f} kWh\n")

    plotting.plot(env)


# %%  ==================  HELP  ========================================

else:
    parser.print_help()
