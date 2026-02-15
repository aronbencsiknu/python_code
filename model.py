"""
Model definition for the RL-BEMS agent.

When using stable-baselines3 SAC, the actor-critic networks are built
automatically by SB3.  This file documents the architecture used.

SAC internally creates:
    Actor  : MLP [256, 256] → Gaussian policy → tanh squash → action
    Critic : Twin Q-networks, each MLP [256, 256] → scalar Q-value
    Entropy : auto-tuned temperature α

No manual model instantiation is needed — SB3 handles everything.

To customise the architecture, pass policy_kwargs to SAC:

    from stable_baselines3 import SAC

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        ...
    )
"""
