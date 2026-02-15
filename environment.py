# %%
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# %%
class WeatherModel:
    """
    Simulates outdoor temperature with a daily sinusoidal cycle
    and AR(1) stochastic noise for realistic variation.

    T_out(t) = T_mean + A_daily * sin(2*pi*(t - 8)/24) + noise
    Min at ~2 AM, max at ~2 PM.
    """

    def __init__(self, T_mean=7.0, A_daily=4.0, ar_phi=0.995, ar_sigma=0.02,
                 rng=None):
        self.T_mean = T_mean
        self.A_daily = A_daily
        self.ar_phi = ar_phi
        self.ar_sigma = ar_sigma
        self.noise = 0.0
        self.current_T_out = T_mean
        self.rng = rng  # gymnasium-seeded np_random, or None for global

    def update(self, time_hours):
        """Update and return current outside temperature."""
        T_base = self.T_mean + self.A_daily * math.sin(
            2 * math.pi * (time_hours - 8.0) / 24.0
        )
        rand_val = self.rng.standard_normal() if self.rng is not None else np.random.randn()
        self.noise = self.ar_phi * self.noise + self.ar_sigma * rand_val
        self.current_T_out = T_base + self.noise
        return self.current_T_out

    def reset(self):
        self.noise = 0.0
        self.current_T_out = self.T_mean


# %%
class HeatingCurve:
    """
    Weather-compensated heating curve.

    T_flow_set = K * (T_room_set - T_out) + T_shift
    Clamped to [T_flow_min, T_flow_max].
    Returns 0 when outside temperature exceeds summer_cutoff.
    """

    def __init__(self, K=1.5, T_shift=0.0, T_room_set=21.0,
                 T_flow_min=25.0, T_flow_max=75.0, summer_cutoff=16.0):
        self.K = K
        self.T_shift = T_shift
        self.T_room_set = T_room_set
        self.T_flow_min = T_flow_min
        self.T_flow_max = T_flow_max
        self.summer_cutoff = summer_cutoff

    def compute_setpoint(self, T_out):
        """Compute target flow temperature from outside temperature."""
        if T_out >= self.summer_cutoff:
            return 0.0
        T_flow_set = self.K * (self.T_room_set - T_out) + self.T_shift
        return float(np.clip(T_flow_set, self.T_flow_min, self.T_flow_max))


# %%
class PIController:
    """
    Position-form PI controller for the mixing valve.

    valve_position = Kp * error + Ki * integral(error)
    With anti-windup integral clamping.
    """

    def __init__(self, Kp=5.0, Ki=0.01, integral_limit=500.0):
        self.Kp = Kp
        self.Ki = Ki
        self.integral = 0.0
        self.integral_limit = integral_limit
        self.valve_position = 0.0

    def step(self, error, dt):
        """Update valve position based on flow-temperature error."""
        self.integral += error * dt
        self.integral = float(np.clip(
            self.integral, -self.integral_limit, self.integral_limit
        ))
        self.valve_position = self.Kp * error + self.Ki * self.integral
        self.valve_position = float(np.clip(self.valve_position, 0.0, 100.0))
        return self.valve_position

    def reset(self):
        self.integral = 0.0
        self.valve_position = 0.0


# %%
class HydronicLoop:
    """
    Hydronic water loop: boiler -> mixing valve -> radiators -> return.

    dT_flow/dt = (Q_valve - Q_emit) / C_water
    Q_emit     = UA_radiator * max(T_flow - T_air, 0)
    Q_valve    = (valve_position / 100) * Q_boiler_max
    T_return   = T_flow - Q_emit / mdot_cp
    """

    def __init__(self, Q_boiler_max=24000.0, UA_radiator=300.0,
                 C_water=200000.0, mdot_cp=500.0):
        self.Q_boiler_max = Q_boiler_max
        self.UA_radiator = UA_radiator
        self.C_water = C_water
        self.mdot_cp = mdot_cp
        self.T_flow = 30.0
        self.T_return = 25.0

    def step(self, valve_position, T_air, dt):
        """Advance hydronic loop by one internal timestep. Returns Q_emit."""
        Q_valve = (valve_position / 100.0) * self.Q_boiler_max
        Q_emit = self.UA_radiator * max(self.T_flow - T_air, 0.0)

        dT_flow = (Q_valve - Q_emit) / self.C_water * dt
        self.T_flow += dT_flow
        self.T_flow = float(np.clip(self.T_flow, 10.0, 90.0))

        if self.mdot_cp > 0:
            self.T_return = self.T_flow - Q_emit / self.mdot_cp
            self.T_return = max(self.T_return, T_air)
        else:
            self.T_return = T_air

        return Q_emit

    def reset(self, T_initial=30.0):
        self.T_flow = T_initial
        self.T_return = T_initial - 5.0


# %%
class Building:
    """
    2nd-order RC building thermal model with separate air and wall nodes.

    Air node:
        C_air * dT_air/dt = Q_emit
                           + UA_wall_air * (T_wall - T_air)
                           - UA_env      * (T_air  - T_out)

    Wall / thermal-mass node:
        C_wall * dT_wall/dt = UA_wall_air * (T_air - T_wall)
    """

    def __init__(self, C_air=500_000.0, C_wall=15_000_000.0,
                 UA_env=200.0, UA_wall_air=800.0):
        self.C_air = C_air
        self.C_wall = C_wall
        self.UA_env = UA_env
        self.UA_wall_air = UA_wall_air
        self.T_air = 20.0
        self.T_wall = 19.0

    def step(self, Q_emit, T_out, dt):
        """Advance building model by one internal timestep (Forward Euler)."""
        Q_wall_to_air = self.UA_wall_air * (self.T_wall - self.T_air)
        Q_loss = self.UA_env * (self.T_air - T_out)

        dT_air = (Q_emit + Q_wall_to_air - Q_loss) / self.C_air * dt
        dT_wall = self.UA_wall_air * (self.T_air - self.T_wall) / self.C_wall * dt

        self.T_air += dT_air
        self.T_wall += dT_wall

        self.T_air = float(np.clip(self.T_air, -10.0, 50.0))
        self.T_wall = float(np.clip(self.T_wall, -10.0, 50.0))

    def reset(self, T_air=20.0, T_wall=19.0):
        self.T_air = T_air
        self.T_wall = T_wall


# %%
class HydronicHeatingEnv(gym.Env):
    """
    Gymnasium-compatible RL environment for a hydronic-heated house
    with weather-compensated control and SAC-ready continuous action space.

    System architecture (5 layers):
        WeatherModel  ->  HeatingCurve (K + shift)
                              |
                       PI Controller  ->  valve_position
                              |
                       HydronicLoop   ->  Q_emit
                              |
                         Building     ->  T_air, T_wall

    Action  : continuous ΔT_shift in [-shift_step_limit, +shift_step_limit]
    State   : 12-dim normalised observation
    Reward  : comfort + energy + smoothness + valve penalties (built-in)
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = {}

        # --- time ---
        self.dt = config.get('dt', 60)
        self.agent_interval = config.get('agent_interval', 15)
        self.episode_days = config.get('episode_days', 5)

        # --- action limits ---
        self.shift_step_limit = config.get('shift_step_limit', 0.5)
        self.shift_limit = config.get('shift_limit', 5.0)

        # --- reward weights ---
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 0.1)
        self.gamma_r = config.get('gamma_r', 0.5)    # "_r" to avoid clash with discount γ
        self.delta = config.get('delta', 0.1)
        self.asymmetry = config.get('asymmetry', 1.5)
        self.hard_band = config.get('hard_band', 2.0)
        self.hard_penalty = config.get('hard_penalty', 5.0)

        # --- domain-randomisation ranges ---
        self.randomize = config.get('randomize', True)
        self.UA_env_range = config.get('UA_env_range', (140.0, 280.0))
        self.C_wall_range = config.get('C_wall_range', (8_000_000.0, 25_000_000.0))
        self.UA_radiator_range = config.get('UA_radiator_range', (200.0, 400.0))
        self.T_mean_range = config.get('T_mean_range', (0.0, 12.0))
        self.A_daily_range = config.get('A_daily_range', (2.0, 7.0))
        self.T_air_init_range = config.get('T_air_init_range', (16.0, 22.0))
        self.K_init_range = config.get('K_init_range', (1.0, 2.0))
        self.T_shift_init_range = config.get('T_shift_init_range', (-2.0, 2.0))
        self.episode_days_range = config.get('episode_days_range', (3, 7))

        self.default_config = config

        # --- Gymnasium spaces ---
        # Continuous action: ΔT_shift in [-0.5, +0.5]
        self.action_space = spaces.Box(
            low=-self.shift_step_limit,
            high=self.shift_step_limit,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation: 12-dim, mostly [0, 1] with sin/cos in [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0,  # generous bounds for normalised values
            shape=(12,),
            dtype=np.float32,
        )

        # --- sub-models (set properly in reset()) ---
        self.weather = WeatherModel()
        self.heating_curve = HeatingCurve()
        self.pi_controller = PIController()
        self.hydronic = HydronicLoop()
        self.building = Building()

        # --- bookkeeping ---
        self.time_seconds = 0
        self.total_steps = 0
        self.agent_steps = 0
        self.max_agent_steps = 0
        self.energy_cumulative = 0.0
        self.prev_valve = 0.0
        self.prev_T_shift = 0.0

        # --- history (every internal step, for plotting) ---
        self._init_history()

    # ------------------------------------------------------------------
    def _init_history(self):
        self.H_T_out = []
        self.H_T_in = []
        self.H_T_wall = []
        self.H_T_flow = []
        self.H_T_flow_set = []
        self.H_T_return = []
        self.H_valve = []
        self.H_T_shift = []
        self.H_K = []
        self.H_Q_boiler = []
        self.H_Q_emit = []
        self.H_energy_cumulative = []
        self.H_time_hours = []
        self.H_T_room_set = []

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Reset for a new episode with domain randomisation."""
        super().reset(seed=seed)

        if self.randomize:
            T_mean = self.np_random.uniform(*self.T_mean_range)
            A_daily = self.np_random.uniform(*self.A_daily_range)
            UA_env = self.np_random.uniform(*self.UA_env_range)
            C_wall = self.np_random.uniform(*self.C_wall_range)
            UA_radiator = self.np_random.uniform(*self.UA_radiator_range)
            T_air_init = self.np_random.uniform(*self.T_air_init_range)
            K_init = self.np_random.uniform(*self.K_init_range)
            T_shift_init = self.np_random.uniform(*self.T_shift_init_range)
            self.episode_days = int(self.np_random.integers(
                self.episode_days_range[0], self.episode_days_range[1] + 1
            ))
        else:
            T_mean = self.default_config.get('T_mean', 7.0)
            A_daily = self.default_config.get('A_daily', 4.0)
            UA_env = self.default_config.get('UA_env', 200.0)
            C_wall = self.default_config.get('C_wall', 15_000_000.0)
            UA_radiator = self.default_config.get('UA_radiator', 300.0)
            T_air_init = self.default_config.get('T_air_init', 20.0)
            K_init = self.default_config.get('K_init', 1.5)
            T_shift_init = self.default_config.get('T_shift_init', 0.0)
            self.episode_days = self.default_config.get('episode_days', 5)

        # sub-models
        self.weather = WeatherModel(T_mean=T_mean, A_daily=A_daily,
                                    rng=self.np_random)
        self.weather.update(0.0)

        self.heating_curve = HeatingCurve(K=K_init, T_shift=T_shift_init)

        self.pi_controller = PIController()
        self.pi_controller.reset()

        self.hydronic = HydronicLoop(UA_radiator=UA_radiator)
        self.hydronic.reset(T_initial=T_air_init + 10.0)

        self.building = Building(UA_env=UA_env, C_wall=C_wall)
        self.building.reset(T_air=T_air_init, T_wall=T_air_init - 1.0)

        # time
        self.time_seconds = 0
        self.total_steps = 0
        self.agent_steps = 0
        agent_interval_minutes = self.agent_interval * self.dt / 60.0
        total_minutes = self.episode_days * 24 * 60
        self.max_agent_steps = int(total_minutes / agent_interval_minutes)

        self.energy_cumulative = 0.0
        self.prev_valve = 0.0
        self.prev_T_shift = self.heating_curve.T_shift

        self._init_history()

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        """
        Execute one agent decision step.

        Parameters
        ----------
        action : np.ndarray, shape (1,)
            Continuous ΔT_shift value in [-shift_step_limit, +shift_step_limit].

        Returns
        -------
        obs, reward, terminated, truncated, info
        """

        # --- apply continuous curve-shift action ---
        delta_shift = float(np.clip(
            action[0], -self.shift_step_limit, self.shift_step_limit
        ))
        self.prev_T_shift = self.heating_curve.T_shift
        self.heating_curve.T_shift += delta_shift
        self.heating_curve.T_shift = float(np.clip(
            self.heating_curve.T_shift, -self.shift_limit, self.shift_limit
        ))

        # --- run internal sub-steps ---
        total_energy_J = 0.0
        valve_oscillation = 0.0
        prev_valve_internal = self.pi_controller.valve_position

        for _ in range(self.agent_interval):
            time_hours = self.time_seconds / 3600.0

            # 1. weather
            T_out = self.weather.update(time_hours)

            # 2. heating curve → flow setpoint
            T_flow_set = self.heating_curve.compute_setpoint(T_out)

            # 3. PI controller → valve
            if T_flow_set > 0:
                error = T_flow_set - self.hydronic.T_flow
                valve = self.pi_controller.step(error, self.dt)
            else:
                valve = 0.0
                self.pi_controller.reset()

            # 4. hydronic loop → heat emission
            Q_emit = self.hydronic.step(valve, self.building.T_air, self.dt)

            # 5. building → temperatures
            self.building.step(Q_emit, T_out, self.dt)

            # energy
            Q_boiler = (valve / 100.0) * self.hydronic.Q_boiler_max
            energy_J = Q_boiler * self.dt
            total_energy_J += energy_J
            self.energy_cumulative += energy_J / 3_600_000.0

            # valve oscillation
            valve_oscillation += abs(
                self.pi_controller.valve_position - prev_valve_internal
            )
            prev_valve_internal = self.pi_controller.valve_position

            # history
            self.H_T_out.append(T_out)
            self.H_T_in.append(self.building.T_air)
            self.H_T_wall.append(self.building.T_wall)
            self.H_T_flow.append(self.hydronic.T_flow)
            self.H_T_flow_set.append(T_flow_set)
            self.H_T_return.append(self.hydronic.T_return)
            self.H_valve.append(valve)
            self.H_T_shift.append(self.heating_curve.T_shift)
            self.H_K.append(self.heating_curve.K)
            self.H_Q_boiler.append(Q_boiler)
            self.H_Q_emit.append(Q_emit)
            self.H_energy_cumulative.append(self.energy_cumulative)
            self.H_time_hours.append(time_hours)
            self.H_T_room_set.append(self.heating_curve.T_room_set)

            self.time_seconds += self.dt
            self.total_steps += 1

        # --- end of agent step ---
        self.agent_steps += 1
        self.prev_valve = self.pi_controller.valve_position
        energy_kWh = total_energy_J / 3_600_000.0
        actual_delta = abs(self.heating_curve.T_shift - self.prev_T_shift)

        # --- compute reward ---
        reward = self._compute_reward(
            self.building.T_air,
            self.heating_curve.T_room_set,
            energy_kWh,
            actual_delta,
            valve_oscillation,
        )

        # --- termination ---
        terminated = False
        truncated = self.agent_steps >= self.max_agent_steps

        obs = self._get_obs()
        info = {
            'T_in': self.building.T_air,
            'T_out': self.weather.current_T_out,
            'T_flow': self.hydronic.T_flow,
            'valve': self.pi_controller.valve_position,
            'T_shift': self.heating_curve.T_shift,
            'energy_kWh': energy_kWh,
            'energy_cumulative': self.energy_cumulative,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _compute_reward(self, T_in, T_set, energy, delta_shift, valve_osc):
        """Compute scalar reward (built into the environment)."""
        temp_error = T_in - T_set

        # quadratic comfort penalty (asymmetric — colder hurts more)
        if temp_error < 0:
            r_comfort = -self.alpha * self.asymmetry * temp_error ** 2
        else:
            r_comfort = -self.alpha * temp_error ** 2

        # hard-band penalty
        if abs(temp_error) > self.hard_band:
            r_comfort -= self.hard_penalty * (abs(temp_error) - self.hard_band)

        r_energy = -self.beta * energy
        r_smooth = -self.gamma_r * delta_shift
        r_valve = -self.delta * (valve_osc / 100.0)

        return float(r_comfort + r_energy + r_smooth + r_valve)

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Build the normalised 12-dim observation vector."""
        time_hours = self.time_seconds / 3600.0
        hour_of_day = time_hours % 24.0
        hour_angle = 2.0 * math.pi * hour_of_day / 24.0

        T_out = self.weather.current_T_out

        obs = np.array([
            (T_out - (-10.0)) / 40.0,                           # T_out
            (self.building.T_air - 10.0) / 25.0,                # T_in
            (self.hydronic.T_flow - 10.0) / 80.0,               # T_flow
            (self.hydronic.T_return - 10.0) / 55.0,             # T_return
            self.pi_controller.valve_position / 100.0,           # valve
            (self.heating_curve.K - 0.5) / 2.5,                 # K
            (self.heating_curve.T_shift + 5.0) / 10.0,          # T_shift
            (self.heating_curve.T_room_set - 18.0) / 6.0,       # T_room_set
            (self.building.T_wall - 10.0) / 25.0,               # T_wall
            math.sin(hour_angle),                                # sin(hour)
            math.cos(hour_angle),                                # cos(hour)
            self.pi_controller.valve_position / 100.0,           # power proxy
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    def get_day_hour_minute(self):
        total_minutes = self.time_seconds / 60.0
        day = int(total_minutes // 1440)
        hour = int((total_minutes % 1440) // 60)
        minute = int(total_minutes % 60)
        return day, hour, minute
