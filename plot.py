import matplotlib.pyplot as plt
import numpy as np


class Plot:
    """
    Multi-panel plotting for the hydronic-heating RL environment.

    Subplots
    --------
    1. Room temperatures : T_out, T_in, T_wall, T_room_set
    2. Flow loop         : T_flow, T_flow_set, T_return
    3. Valve position    : 0–100 %
    4. Heating-curve shift & slope K
    5. Energy            : instantaneous boiler power (kW) + cumulative (kWh)
    """

    def __init__(self):
        self.c_T_out = "royalblue"
        self.c_T_in = "orangered"
        self.c_T_wall = "sandybrown"
        self.c_T_set = "grey"
        self.c_T_flow = "crimson"
        self.c_T_flow_set = "salmon"
        self.c_T_return = "steelblue"
        self.c_valve = "seagreen"
        self.c_shift = "darkorchid"
        self.c_K = "teal"
        self.c_power = "coral"
        self.c_energy = "goldenrod"

    # ------------------------------------------------------------------
    def plot(self, env):
        """Plot a full episode dashboard from an Environment instance."""

        if len(env.H_time_hours) == 0:
            print("No history to plot.")
            return

        t = np.array(env.H_time_hours)

        fig, axes = plt.subplots(5, 1, figsize=(18, 16), sharex=True)
        fig.suptitle("Hydronic Heating RL-BEMS  –  Episode Dashboard",
                     fontsize=14, fontweight="bold")

        # ----- 1. Room temperatures -----
        ax = axes[0]
        ax.plot(t, env.H_T_out, label="T_out", lw=1, color=self.c_T_out)
        ax.plot(t, env.H_T_in, label="T_in (air)", lw=1.5, color=self.c_T_in)
        ax.plot(t, env.H_T_wall, label="T_wall", lw=1, color=self.c_T_wall,
                linestyle="--")
        ax.plot(t, env.H_T_room_set, label="T_room_set", lw=1,
                color=self.c_T_set, linestyle=":")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # ----- 2. Flow loop -----
        ax = axes[1]
        ax.plot(t, env.H_T_flow, label="T_flow", lw=1.2, color=self.c_T_flow)
        ax.plot(t, env.H_T_flow_set, label="T_flow_set", lw=1,
                color=self.c_T_flow_set, linestyle="--")
        ax.plot(t, env.H_T_return, label="T_return", lw=1,
                color=self.c_T_return)
        ax.set_ylabel("Temperature (°C)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # ----- 3. Valve position -----
        ax = axes[2]
        ax.plot(t, env.H_valve, label="Valve %", lw=1, color=self.c_valve)
        ax.set_ylabel("Valve (%)")
        ax.set_ylim(-5, 105)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # ----- 4. Heating-curve shift & K -----
        ax = axes[3]
        ax.plot(t, env.H_T_shift, label="T_shift (K)", lw=1.2,
                color=self.c_shift)
        ax2 = ax.twinx()
        ax2.plot(t, env.H_K, label="Slope K", lw=1, color=self.c_K,
                 linestyle="--")
        ax.set_ylabel("Curve shift (K)")
        ax2.set_ylabel("Slope K")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # ----- 5. Energy -----
        ax = axes[4]
        power_kW = np.array(env.H_Q_boiler) / 1000.0
        ax.plot(t, power_kW, label="Boiler power (kW)", lw=1,
                color=self.c_power, alpha=0.7)
        ax.set_ylabel("Power (kW)")

        ax2 = ax.twinx()
        ax2.plot(t, env.H_energy_cumulative, label="Cumulative (kWh)",
                 lw=1.5, color=self.c_energy)
        ax2.set_ylabel("Energy (kWh)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # shared x-axis
        axes[-1].set_xlabel("Time (hours)")
        self._format_x_ticks(axes[-1], t)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    @staticmethod
    def _format_x_ticks(ax, t):
        """Show day boundaries and 6-hour marks."""
        if len(t) == 0:
            return
        max_h = t[-1]
        major = np.arange(0, max_h + 1, 24)
        minor = np.arange(0, max_h + 1, 6)
        ax.set_xticks(major)
        ax.set_xticks(minor, minor=True)
        labels = [f"Day {int(h // 24)}" for h in major]
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='minor', length=3)

    # ------------------------------------------------------------------
    @staticmethod
    def validate(env_class, config=None):
        """
        Run two open-loop validation tests:
          1. Step test  – increase shift by +1 K, observe indoor response.
          2. Cold-snap  – drop outside temp, observe response.

        Uses continuous actions (np arrays).
        """
        if config is None:
            config = {
                'randomize': False, 'episode_days': 3,
                'dt': 60, 'agent_interval': 15,
            }

        def run_episode(env, shift_schedule):
            """shift_schedule: dict {agent_step: float ΔT_shift}"""
            obs, _ = env.reset()
            step_idx = 0
            done = False
            while not done:
                if step_idx in shift_schedule:
                    action = np.array([shift_schedule[step_idx]], dtype=np.float32)
                else:
                    action = np.array([0.0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                step_idx += 1
                done = terminated or truncated

        # ---- test 1: step test (+0.5 K twice = +1 K at ~12 h) ----
        env1 = env_class(config)
        run_episode(env1, {48: 0.5, 49: 0.5})

        # ---- test 2: cold snap ----
        cold_config = dict(config)
        cold_config['T_mean'] = -1.0
        cold_config['A_daily'] = 2.0
        env2 = env_class(cold_config)
        run_episode(env2, {})

        # ---- plot both ----
        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle("Validation Tests", fontsize=13, fontweight="bold")

        for col, (env, title) in enumerate([
            (env1, "Step Test (+1 K shift at hour 12)"),
            (env2, "Cold-Snap Test (T_mean = -1 °C)"),
        ]):
            t = np.array(env.H_time_hours)

            ax = axes[0, col]
            ax.plot(t, env.H_T_out, label="T_out", lw=1, color="royalblue")
            ax.plot(t, env.H_T_in, label="T_in", lw=1.5, color="orangered")
            ax.plot(t, env.H_T_room_set, label="T_set", lw=1, color="grey",
                    linestyle=":")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title(title)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            ax = axes[1, col]
            ax.plot(t, env.H_T_flow, label="T_flow", lw=1, color="crimson")
            ax.plot(t, env.H_T_flow_set, label="T_flow_set", lw=1,
                    color="salmon", linestyle="--")
            ax.plot(t, env.H_valve, label="Valve %", lw=1, color="seagreen",
                    alpha=0.6)
            ax.set_ylabel("T (°C) / Valve (%)")
            ax.set_xlabel("Time (hours)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
