ENV_KWARGS = None
def set_env_kwargs(d: dict):
    required = {"hmax","initial_amount","num_stock_shares","buy_cost_pct","sell_cost_pct",
                "state_space","stock_dim","tech_indicator_list","action_space","reward_scaling"}
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"set_env_kwargs missing keys: {missing}")
    global ENV_KWARGS
    ENV_KWARGS = dict(d)

import gymnasium as gym

def trade_risk_rppo(
    model_path_base,
    trade_df,
    loop_count=5,
    # risk/eval knobs (tweak if you trained with different ones)
    risk_metric="sharpe",          # 'sharpe' | 'sortino' | 'pnl'
    window=63,
    annualization=252.0,
    scale=1.0,
    turbulence_threshold=55,
    risk_indicator_col="vix",
):
    """
    RPPO + risk-aware reward (evaluation wrapper).
    Returns a list of account-value curves (one per run).
    """
    import os, glob, numpy as np
    import gymnasium as gym
    from collections import deque
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    # --- env import (prefer your local env) ---
    try:
        from env_stocktrading import StockTradingEnv
    except Exception:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    # --- RPPO class ---
    try:
        from sb3_contrib import RecurrentPPO
    except Exception as e:
        raise ImportError("RecurrentPPO not found. Install sb3-contrib: pip install sb3-contrib") from e

    # --- pull shared env kwargs (from set_env_kwargs) or infer from globals ---
    def _fetch_env_kwargs():
        g = globals()
        if "ENV_KWARGS" in g and g["ENV_KWARGS"] is not None:
            return dict(g["ENV_KWARGS"])
        needed = {"num_stock_shares", "buy_cost_list", "sell_cost_list",
                  "state_space", "stock_dimension", "INDICATORS"}
        missing = [k for k in needed if k not in g]
        if missing:
            raise RuntimeError(
                "Env kwargs not configured. Either call set_env_kwargs({...}) once, "
                f"or define these globals before importing trading_dispatch: {missing}"
            )
        return {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "num_stock_shares": g["num_stock_shares"],
            "buy_cost_pct": g["buy_cost_list"],
            "sell_cost_pct": g["sell_cost_list"],
            "state_space": g["state_space"],
            "stock_dim": g["stock_dimension"],
            "tech_indicator_list": g["INDICATORS"],
            "action_space": g["stock_dimension"],
            "reward_scaling": 1e-4,
        }

    env_kwargs = _fetch_env_kwargs()

    # ---------- Adapter: force Gymnasium API regardless of underlying env ----------
    class GymAPIAdapter(gym.Wrapper):
        """Coerce reset->(obs, info) and step->(obs, reward, terminated, truncated, info).
           Also tolerates envs that don't accept seed/options in reset()."""
        def reset(self, **kwargs):
            try:
                out = self.env.reset(**kwargs)
            except TypeError:
                # underlying reset doesn't accept seed/options
                out = self.env.reset()
            if isinstance(out, tuple):
                if len(out) >= 2:
                    obs = out[0]
                    info = out[1] if isinstance(out[1], dict) else {}
                    return obs, info
                elif len(out) == 1:
                    return out[0], {}
                else:
                    return None, {}
            # single object
            return out, {}

        def step(self, action):
            out = self.env.step(action)
            if isinstance(out, tuple):
                if len(out) == 5:
                    return out  # already Gymnasium
                if len(out) == 4:
                    obs, reward, done, info = out
                    return obs, reward, bool(done), False, info
                # best-effort fallback
                obs = out[0] if len(out) > 0 else None
                reward = float(out[1]) if len(out) > 1 else 0.0
                done = bool(out[2]) if len(out) > 2 else False
                info = out[-1] if len(out) and isinstance(out[-1], dict) else {}
                return obs, reward, done, False, info
            raise RuntimeError(f"Unexpected step() return type: {type(out)}")

    # ---------- Risk-aware reward wrapper on normalized API ----------
    class RiskRewardWrapper(gym.Wrapper):
        def __init__(self, env, risk_metric="sharpe", window=63, annualization=252.0, scale=1.0):
            super().__init__(env)
            self.risk_metric = risk_metric
            self.window = window
            self.annualization = annualization
            self.scale = scale
            self._returns = deque(maxlen=window)
            self._last_value = None
            self.eps = 1e-8

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)   # guaranteed (obs, info)
            self._returns.clear()
            self._last_value = self._portfolio_value()
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)  # guaranteed 5-tuple
            current_val = self._portfolio_value()
            ret = (current_val - self._last_value) / (self._last_value + self.eps)
            self._last_value = current_val
            self._returns.append(ret)

            if self.risk_metric == "pnl" or len(self._returns) < 2:
                shaped = ret
            elif self.risk_metric == "sharpe":
                mean, std = np.mean(self._returns), np.std(self._returns) + self.eps
                shaped = (mean / std) * np.sqrt(self.annualization)
            elif self.risk_metric == "sortino":
                mean = np.mean(self._returns)
                downside = np.std([r for r in self._returns if r < 0] or [0.0]) + self.eps
                shaped = (mean / downside) * np.sqrt(self.annualization)
            else:
                shaped = ret

            return obs, float(shaped * self.scale), bool(terminated), bool(truncated), info

        def _portfolio_value(self):
            state = self.env.state
            stock_dim = self.env.stock_dim
            prices = np.array(state[1:1 + stock_dim])
            shares = np.array(state[1 + stock_dim:1 + 2 * stock_dim])
            return float(state[0] + (prices * shares).sum())

    # ---------- Final shim: guarantee outermost reset returns exactly (obs, info) ----------
    class FinalResetShim(gym.Wrapper):
        def reset(self, **kwargs):
            try:
                out = self.env.reset(**kwargs)
            except TypeError:
                out = self.env.reset()
            # Normalize ONE LAST TIME
            if isinstance(out, tuple):
                if len(out) >= 2 and isinstance(out[1], dict):
                    return out[0], out[1]
                else:
                    obs = out[0] if len(out) else None
                    info = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
                    return obs, info
            return out, {}

    # ---------- Build eval VecEnv factory ----------
    def make_eval_env():
        try:
            base_env = StockTradingEnv(
                df=trade_df,
                turbulence_threshold=turbulence_threshold,
                risk_indicator_col=risk_indicator_col,
                **env_kwargs
            )
        except TypeError:
            base_env = StockTradingEnv(df=trade_df, **env_kwargs)

        env = GymAPIAdapter(base_env)     # normalize API first
        env = Monitor(env)                # SB3-recommended
        env = RiskRewardWrapper(env, risk_metric=risk_metric, window=window,
                                annualization=annualization, scale=scale)
        env = FinalResetShim(env)         # absolutely ensure reset -> (obs, info)
        return env

    # ---------- helper to run one model ----------
    def _run_one(model_file):
        if not os.path.isfile(model_file):
            return None

        eval_env = DummyVecEnv([make_eval_env])  # VecEnv expects reset -> (obs, info)
        env0 = eval_env.envs[0].unwrapped
        model = RecurrentPPO.load(model_file)

        obs = eval_env.reset()  # VecEnv.reset returns obs only
        lstm_states = None
        episode_start = np.ones((eval_env.num_envs,), dtype=bool)

        stock_dim = getattr(env0, "stock_dim", env_kwargs.get("stock_dim"))
        prev_date = None
        values = []

        while True:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            obs, rewards, dones, infos = eval_env.step(action)
            episode_start = dones

            state = env0.state
            cash = state[0]
            prices = np.array(state[1:1 + stock_dim])
            shares = np.array(state[1 + stock_dim:1 + 2 * stock_dim])
            total_asset = float(cash + (prices * shares).sum())
            cur_date = env0.date_memory[-1] if hasattr(env0, "date_memory") else None

            if prev_date is not None and cur_date is not None and cur_date < prev_date:
                break

            values.append(total_asset)
            prev_date = cur_date

            if bool(np.array(dones).any()):
                break

        if values:
            arr = np.asarray(values, dtype=float)
            rets = np.diff(arr) / (arr[:-1] + 1e-8)
            sharpe = (np.sqrt(252) * np.nanmean(rets) / (np.nanstd(rets) + 1e-8)) if len(rets) > 1 else 0.0
            print(f"[RiskRPPO] {os.path.basename(model_file)} — Final: {arr[-1]:.2f} | Sharpe: {sharpe:.4f}")
        return values if values else None

    # ---------- locate model files and run ----------
    curves = []
    for i in range(loop_count):
        candidates = [os.path.join(model_path_base, f"RiskRPPO_5k_{i}.zip")]
        if not any(os.path.isfile(c) for c in candidates):
            globs = [
                os.path.join(model_path_base, f"*Risk*RPPO*{i}*.zip"),
                os.path.join(model_path_base, f"*RPPO*Risk*{i}*.zip"),
                os.path.join(model_path_base, f"*rppo*risk*{i}*.zip"),
            ]
            matches = []
            for gpat in globs:
                matches.extend(glob.glob(gpat))
            if matches:
                candidates = [sorted(matches)[0]]

        chosen = next((c for c in candidates if os.path.isfile(c)), None)
        if chosen is None:
            print(f"[RiskRPPO] Skip run {i}: no model file in {model_path_base}")
            continue

        vals = _run_one(chosen)
        if vals is not None and np.isfinite(np.asarray(vals)).any():
            curves.append(vals)
        else:
            print(f"[RiskRPPO] Run {i}: empty/NaN curve")

    return curves

def trade_metappo(
    model_path_base,
    trade_df,
    loop_count=5,
    turbulence_threshold=55.0,
    risk_indicator_col="vix",
):
    """
    Meta-PPO trading runner for the dispatcher.
    Returns a list of account-value curves (one per run).
    """
    import os, glob, numpy as np

    # --- env import (prefer your local env) ---
    try:
        from env_stocktrading import StockTradingEnv
    except Exception:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    # --- SB3 PPO ---
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise ImportError("PPO not found. Install stable-baselines3.") from e

    # --- pull shared env kwargs (from set_env_kwargs) or infer from globals ---
    def _fetch_env_kwargs():
        g = globals()
        if "ENV_KWARGS" in g and g["ENV_KWARGS"] is not None:
            return dict(g["ENV_KWARGS"])
        needed = {
            "num_stock_shares", "buy_cost_list", "sell_cost_list",
            "state_space", "stock_dimension", "INDICATORS"
        }
        missing = [k for k in needed if k not in g]
        if missing:
            raise RuntimeError(
                "Env kwargs not configured. Either call set_env_kwargs({...}) once, "
                f"or define these globals before importing trading_dispatch: {missing}"
            )
        return {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "num_stock_shares": g["num_stock_shares"],
            "buy_cost_pct": g["buy_cost_list"],
            "sell_cost_pct": g["sell_cost_list"],
            "state_space": g["state_space"],
            "stock_dim": g["stock_dimension"],
            "tech_indicator_list": g["INDICATORS"],
            "action_space": g["stock_dimension"],
            "reward_scaling": 1e-4,
        }

    env_kwargs = _fetch_env_kwargs()

    # --- build eval env (no VecEnv needed) ---
    def _make_env():
        try:
            return StockTradingEnv(
                df=trade_df,
                turbulence_threshold=turbulence_threshold,
                risk_indicator_col=risk_indicator_col,
                **env_kwargs
            )
        except TypeError:
            # If your env doesn't accept turbulence/risk args
            return StockTradingEnv(df=trade_df, **env_kwargs)

    def _portfolio_value_from_env(e):
        # Prefer direct attribute if available
        for k in ("portfolio_value", "account_value", "total_asset", "nav"):
            if hasattr(e, k):
                try:
                    return float(getattr(e, k))
                except Exception:
                    pass
        # Fallback: compute from state [cash, prices..., shares...]
        s = e.state
        sd = e.stock_dim
        cash = float(s[0])
        prices = np.array(s[1:1+sd], dtype=float)
        shares = np.array(s[1+sd:1+2*sd], dtype=float)
        return float(cash + (prices * shares).sum())

    def _run_one(model_file):
        if not os.path.isfile(model_file):
            return None

        env = _make_env()
        try:
            model = PPO.load(model_file, env=env, device="auto")
        except Exception as e:
            # Try loading without binding env (SB3 allows predict without env)
            model = PPO.load(model_file, device="auto")

        # reset handling for gym / gymnasium
        out = env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
        else:
            obs = out

        values = []
        prev_date = None
        while True:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            # gymnasium: (obs, reward, terminated, truncated, info)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out

            total_asset = _portfolio_value_from_env(env)
            cur_date = env.date_memory[-1] if hasattr(env, "date_memory") and env.date_memory else None

            # auto-reset guard: if date goes backwards, break before logging
            if prev_date is not None and cur_date is not None and cur_date < prev_date:
                break

            values.append(total_asset)
            prev_date = cur_date

            if done:
                break

        if values:
            arr = np.asarray(values, dtype=float)
            rets = np.diff(arr) / (arr[:-1] + 1e-8)
            sharpe = (np.sqrt(252) * np.nanmean(rets) / (np.nanstd(rets) + 1e-8)) if len(rets) > 1 else 0.0
            print(f"[MetaPPO] {os.path.basename(model_file)} — Final: {arr[-1]:.2f} | Sharpe: {sharpe:.4f}")
        return values if values else None

    # --- locate model files + run ---
    curves = []
    for i in range(loop_count):
        candidates = [os.path.join(model_path_base, f"MetaPPO_5k_{i}.zip")]
        if not any(os.path.isfile(c) for c in candidates):
            # Flexible fallbacks
            globs = [
                os.path.join(model_path_base, f"*Meta*PPO*{i}*.zip"),
                os.path.join(model_path_base, f"*PPO*Meta*{i}*.zip"),
                os.path.join(model_path_base, f"*metappo*{i}*.zip"),
            ]
            matches = []
            for gpat in globs:
                matches.extend(glob.glob(gpat))
            if matches:
                candidates = [sorted(matches)[0]]

        chosen = next((c for c in candidates if os.path.isfile(c)), None)
        if chosen is None:
            print(f"[MetaPPO] Skip run {i}: no model file in {model_path_base}")
            continue

        vals = _run_one(chosen)
        if vals is not None and np.isfinite(np.asarray(vals)).any():
            curves.append(vals)
        else:
            print(f"[MetaPPO] Run {i}: empty/NaN curve")

    return curves

def trade_rppo(
    model_path_base,
    trade_df,
    loop_count=5,
):
    """
    Recurrent PPO (RPPO) trading runner for the dispatcher.
    Returns a list of account-value curves (one per run).
    """
    import os, glob, numpy as np

    # --- env import (prefer your local env) ---
    try:
        from env_stocktrading import StockTradingEnv
    except Exception:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    # --- sb3-contrib RPPO ---
    try:
        from sb3_contrib import RecurrentPPO
    except Exception as e:
        raise ImportError("RecurrentPPO not found. Install sb3-contrib: pip install sb3-contrib") from e

    # Vec env (RPPO plays nicest with a VecEnv wrapper)
    from stable_baselines3.common.vec_env import DummyVecEnv

    # --- pull shared env kwargs (from set_env_kwargs) or infer from globals ---
    def _fetch_env_kwargs():
        g = globals()
        if "ENV_KWARGS" in g and g["ENV_KWARGS"] is not None:
            return dict(g["ENV_KWARGS"])
        needed = {
            "num_stock_shares", "buy_cost_list", "sell_cost_list",
            "state_space", "stock_dimension", "INDICATORS"
        }
        missing = [k for k in needed if k not in g]
        if missing:
            raise RuntimeError(
                "Env kwargs not configured. Either call set_env_kwargs({...}) once, "
                f"or define these globals before importing trading_dispatch: {missing}"
            )
        return {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "num_stock_shares": g["num_stock_shares"],
            "buy_cost_pct": g["buy_cost_list"],
            "sell_cost_pct": g["sell_cost_list"],
            "state_space": g["state_space"],
            "stock_dim": g["stock_dimension"],
            "tech_indicator_list": g["INDICATORS"],
            "action_space": g["stock_dimension"],
            "reward_scaling": 1e-4,
        }

    env_kwargs = _fetch_env_kwargs()

    # --- build eval VecEnv factory (self-contained) ---
    def _make_env():
        return StockTradingEnv(df=trade_df, **env_kwargs)

    def _portfolio_value_from_env(e):
        # Prefer direct attribute if available
        for k in ("portfolio_value", "account_value", "total_asset", "nav"):
            if hasattr(e, k):
                try:
                    return float(getattr(e, k))
                except Exception:
                    pass
        # Fallback: compute from state [cash, prices..., shares...]
        s = e.state
        sd = e.stock_dim
        cash = float(s[0])
        prices = np.array(s[1:1+sd], dtype=float)
        shares = np.array(s[1+sd:1+2*sd], dtype=float)
        return float(cash + (prices * shares).sum())

    def _run_one(model_file):
        if not os.path.isfile(model_file):
            return None

        eval_env = DummyVecEnv([_make_env])
        env0 = eval_env.envs[0].unwrapped

        # Load RPPO model
        model = RecurrentPPO.load(model_file)

        # Reset (Gymnasium or Gym)
        obs = eval_env.reset()
        lstm_states = None
        episode_start = np.ones((eval_env.num_envs,), dtype=bool)

        values = []
        prev_date = None

        while True:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            # DummyVecEnv step: (obs, rewards, dones, infos)
            obs, rewards, dones, infos = eval_env.step(action)
            episode_start = dones

            total_asset = _portfolio_value_from_env(env0)
            cur_date = env0.date_memory[-1] if hasattr(env0, "date_memory") and env0.date_memory else None

            # auto-reset guard: if date goes backwards, exit before logging
            if prev_date is not None and cur_date is not None and cur_date < prev_date:
                break

            values.append(total_asset)
            prev_date = cur_date

            if bool(np.array(dones).any()):
                break

        if values:
            arr = np.asarray(values, dtype=float)
            rets = np.diff(arr) / (arr[:-1] + 1e-8)
            sharpe = (np.sqrt(252) * np.nanmean(rets) / (np.nanstd(rets) + 1e-8)) if len(rets) > 1 else 0.0
            print(f"[RPPO] {os.path.basename(model_file)} — Final: {arr[-1]:.2f} | Sharpe: {sharpe:.4f}")
        return values if values else None

    # --- locate model files + run ---
    curves = []
    for i in range(loop_count):
        candidates = [os.path.join(model_path_base, f"RPPO_5k_{i}.zip")]
        if not any(os.path.isfile(c) for c in candidates):
            # Flexible fallbacks
            globs = [
                os.path.join(model_path_base, f"*RPPO*{i}*.zip"),
                os.path.join(model_path_base, f"*recurrent*ppo*{i}*.zip"),
                os.path.join(model_path_base, f"*rppo*{i}*.zip"),
            ]
            matches = []
            for gpat in globs:
                matches.extend(glob.glob(gpat))
            if matches:
                candidates = [sorted(matches)[0]]

        chosen = next((c for c in candidates if os.path.isfile(c)), None)
        if chosen is None:
            print(f"[RPPO] Skip run {i}: no model file in {model_path_base}")
            continue

        vals = _run_one(chosen)
        if vals is not None and np.isfinite(np.asarray(vals)).any():
            curves.append(vals)
        else:
            print(f"[RPPO] Run {i}: empty/NaN curve")

    return curves


def trade_riskppo(
    model_path_base,
    trade_df,
    loop_count=5,
    turbulence_threshold=55.0,
    risk_indicator_col="vix",
):
    """
    Risk-aware PPO (RiskPPO) trading runner for the dispatcher.
    Returns: List[List[float]] — one equity curve per loaded model.
    """
    import os, glob, numpy as np

    # --- env import (prefer your local env) ---
    try:
        from env_stocktrading import StockTradingEnv
    except Exception:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    # --- SB3 PPO ---
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise ImportError("PPO not found. Install stable-baselines3.") from e

    # --- pull shared env kwargs (from set_env_kwargs) or infer from globals ---
    def _fetch_env_kwargs():
        g = globals()
        if "ENV_KWARGS" in g and g["ENV_KWARGS"] is not None:
            return dict(g["ENV_KWARGS"])
        needed = {
            "num_stock_shares", "buy_cost_list", "sell_cost_list",
            "state_space", "stock_dimension", "INDICATORS"
        }
        missing = [k for k in needed if k not in g]
        if missing:
            raise RuntimeError(
                "Env kwargs not configured. Either call set_env_kwargs({...}) once, "
                f"or define these globals before importing trading_dispatch: {missing}"
            )
        return {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "num_stock_shares": g["num_stock_shares"],
            "buy_cost_pct": g["buy_cost_list"],
            "sell_cost_pct": g["sell_cost_list"],
            "state_space": g["state_space"],
            "stock_dim": g["stock_dimension"],
            "tech_indicator_list": g["INDICATORS"],
            "action_space": g["stock_dimension"],
            "reward_scaling": 1e-4,
        }

    env_kwargs = _fetch_env_kwargs()

    # --- build eval env (no VecEnv required for PPO) ---
    def _make_env():
        try:
            return StockTradingEnv(
                df=trade_df,
                turbulence_threshold=turbulence_threshold,
                risk_indicator_col=risk_indicator_col,
                **env_kwargs
            )
        except TypeError:
            # If your env doesn't accept turbulence/risk args
            return StockTradingEnv(df=trade_df, **env_kwargs)

    def _portfolio_value_from_env(e):
        # Prefer direct attribute if exposed
        for k in ("portfolio_value", "account_value", "total_asset", "nav"):
            if hasattr(e, k):
                try:
                    return float(getattr(e, k))
                except Exception:
                    pass
        # Fallback: compute from state [cash, prices..., shares...]
        s = e.state
        sd = e.stock_dim
        cash = float(s[0])
        prices = np.array(s[1:1+sd], dtype=float)
        shares = np.array(s[1+sd:1+2*sd], dtype=float)
        return float(cash + (prices * shares).sum())

    def _run_one(model_file):
        if not os.path.isfile(model_file):
            return None

        env = _make_env()
        try:
            model = PPO.load(model_file, env=env, device="auto")
        except Exception:
            # SB3 can load without env; bind later via predict
            model = PPO.load(model_file, device="auto")

        # reset handling for gym / gymnasium
        out = env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
        else:
            obs = out

        values = []
        prev_date = None
        while True:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            # gymnasium: (obs, reward, terminated, truncated, info)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out

            total_asset = _portfolio_value_from_env(env)
            cur_date = env.date_memory[-1] if hasattr(env, "date_memory") and env.date_memory else None

            # auto-reset guard: if date goes backwards, stop before logging
            if prev_date is not None and cur_date is not None and cur_date < prev_date:
                break

            values.append(total_asset)
            prev_date = cur_date

            if done:
                break

        if values:
            arr = np.asarray(values, dtype=float)
            rets = np.diff(arr) / (arr[:-1] + 1e-8)
            sharpe = (np.sqrt(252) * np.nanmean(rets) / (np.nanstd(rets) + 1e-8)) if len(rets) > 1 else 0.0
            print(f"[RiskPPO] {os.path.basename(model_file)} — Final: {arr[-1]:.2f} | Sharpe: {sharpe:.4f}")
        return values if values else None

    # --- locate model files + run ---
    curves = []
    for i in range(loop_count):
        candidates = [
            os.path.join(model_path_base, f"RiskPPO_5k_{i}.zip"),
        ]
        # Flexible fallbacks (common names in your standalone setup)
        if not any(os.path.isfile(c) for c in candidates):
            globs = [
                os.path.join(model_path_base, f"*Risk*PPO*{i}*.zip"),
                os.path.join(model_path_base, f"*PPO*Risk*{i}*.zip"),
                os.path.join(model_path_base, f"*riskppo*{i}*.zip"),
            ]
            matches = []
            for gpat in globs:
                matches.extend(glob.glob(gpat))
            # if only one generic file like "trained_riskppo.zip" exists, use it for i==0
            if not matches and i == 0:
                matches = glob.glob(os.path.join(model_path_base, "trained_riskppo*.zip")) \
                       or glob.glob(os.path.join(model_path_base, "*riskppo*.zip"))
            if matches:
                candidates = [sorted(matches)[0]]

        chosen = next((c for c in candidates if os.path.isfile(c)), None)
        if chosen is None:
            print(f"[RiskPPO] Skip run {i}: no model file in {model_path_base}")
            continue

        vals = _run_one(chosen)
        if vals is not None and np.isfinite(np.asarray(vals)).any():
            curves.append(vals)
        else:
            print(f"[RiskPPO] Run {i}: empty/NaN curve")

    return curves

def trade_constrainppo(
    model_path_base,
    trade_df,
    loop_count=5,
    turbulence_threshold=55.0,
    risk_indicator_col="vix",
):
    """
    Constraint-Aware PPO trading runner for the dispatcher.
    Returns: List[List[float]] — one equity curve per loaded model.
    """
    import os, glob, numpy as np

    # --- prefer your local env; fallback to FinRL's ---
    try:
        from env_stocktrading import StockTradingEnv
    except Exception:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    # --- SB3 PPO ---
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise ImportError("PPO not found. Install stable-baselines3.") from e

    # --- shared env kwargs ---
    def _fetch_env_kwargs():
        g = globals()
        if "ENV_KWARGS" in g and g["ENV_KWARGS"] is not None:
            return dict(g["ENV_KWARGS"])
        needed = {
            "num_stock_shares", "buy_cost_list", "sell_cost_list",
            "state_space", "stock_dimension", "INDICATORS"
        }
        missing = [k for k in needed if k not in g]
        if missing:
            raise RuntimeError(
                "Env kwargs not configured. Either call set_env_kwargs({...}) once, "
                f"or define these globals before importing trading_dispatch: {missing}"
            )
        return {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "num_stock_shares": g["num_stock_shares"],
            "buy_cost_pct": g["buy_cost_list"],
            "sell_cost_pct": g["sell_cost_list"],
            "state_space": g["state_space"],
            "stock_dim": g["stock_dimension"],
            "tech_indicator_list": g["INDICATORS"],
            "action_space": g["stock_dimension"],
            "reward_scaling": 1e-4,
        }

    env_kwargs = _fetch_env_kwargs()

    # --- env factory (passes risk args if supported) ---
    def _make_env():
        try:
            return StockTradingEnv(
                df=trade_df,
                turbulence_threshold=turbulence_threshold,
                risk_indicator_col=risk_indicator_col,
                **env_kwargs
            )
        except TypeError:
            return StockTradingEnv(df=trade_df, **env_kwargs)

    def _portfolio_value_from_env(e):
        for k in ("portfolio_value", "account_value", "total_asset", "nav"):
            if hasattr(e, k):
                try:
                    return float(getattr(e, k))
                except Exception:
                    pass
        s = e.state
        sd = e.stock_dim
        cash = float(s[0])
        prices = np.array(s[1:1+sd], dtype=float)
        shares = np.array(s[1+sd:1+2*sd], dtype=float)
        return float(cash + (prices * shares).sum())

    def _run_one(model_file):
        if not os.path.isfile(model_file):
            return None

        env = _make_env()
        try:
            model = PPO.load(model_file, env=env, device="auto")
        except Exception:
            model = PPO.load(model_file, device="auto")

        out = env.reset()
        obs = out[0] if isinstance(out, tuple) and len(out) == 2 else out

        values = []
        prev_date = None

        while True:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if len(step_out) == 5:  # gymnasium
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:                   # gym
                obs, reward, done, info = step_out

            total_asset = _portfolio_value_from_env(env)
            cur_date = env.date_memory[-1] if hasattr(env, "date_memory") and env.date_memory else None

            # auto-reset guard
            if prev_date is not None and cur_date is not None and cur_date < prev_date:
                break

            values.append(total_asset)
            prev_date = cur_date

            if done:
                break

        if values:
            arr = np.asarray(values, dtype=float)
            rets = np.diff(arr) / (arr[:-1] + 1e-8)
            sharpe = (np.sqrt(252) * np.nanmean(rets) / (np.nanstd(rets) + 1e-8)) if len(rets) > 1 else 0.0
            print(f"[ConstrainPPO] {os.path.basename(model_file)} — Final: {arr[-1]:.2f} | Sharpe: {sharpe:.4f}")
        return values if values else None

    # --- find models & run ---
    curves = []
    for i in range(loop_count):
        candidates = [os.path.join(model_path_base, f"ConstrainPPO_5k_{i}.zip")]
        if not any(os.path.isfile(c) for c in candidates):
            globs = [
                os.path.join(model_path_base, f"*Constrain*PPO*{i}*.zip"),
                os.path.join(model_path_base, f"*Constraint*PPO*{i}*.zip"),
                os.path.join(model_path_base, f"*constrainppo*{i}*.zip"),
            ]
            matches = []
            for gpat in globs:
                matches.extend(glob.glob(gpat))
            if not matches and i == 0:
                matches = glob.glob(os.path.join(model_path_base, "*constrain*ppo*.zip"))
            if matches:
                candidates = [sorted(matches)[0]]

        chosen = next((c for c in candidates if os.path.isfile(c)), None)
        if chosen is None:
            print(f"[ConstrainPPO] Skip run {i}: no model file in {model_path_base}")
            continue

        vals = _run_one(chosen)
        if vals is not None and np.isfinite(np.asarray(vals)).any():
            curves.append(vals)
        else:
            print(f"[ConstrainPPO] Run {i}: empty/NaN curve")

    return curves

def trade_transformerppo(
    model_path_base,
    trade_df,
    loop_count=5,
    turbulence_threshold=55.0,
    risk_indicator_col="vix",
):
    """
    Transformer-PPO trading runner for the dispatcher.
    Returns: List[List[float]] — one equity curve per loaded model.
    """
    import os, glob, numpy as np
    import torch as th
    from torch import nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    # --- prefer your local env; fallback to FinRL's ---
    try:
        from env_stocktrading import StockTradingEnv
    except Exception:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    # --- SB3 PPO ---
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise ImportError("PPO not found. Install stable-baselines3.") from e

    # ---------- Transformer features extractor that matches saved checkpoints ----------
    class TransformerFeaturesExtractor(BaseFeaturesExtractor):
        """
        Matches checkpoints that have keys:
          - features_extractor.positional_encoding
          - features_extractor.linear_embed.*
          - features_extractor.transformer.layers.0.*
          - features_extractor.layer_norm.*
        IMPORTANT: features_dim must match training (error showed 32).
        """
        def __init__(self, observation_space, features_dim: int = 32,
                     n_heads: int = 4, n_layers: int = 1, dropout: float = 0.1):
            super().__init__(observation_space, features_dim)
            # flatten obs to a single vector
            obs_shape = observation_space.shape
            obs_dim = int(np.prod(obs_shape))
            self.linear_embed = nn.Linear(obs_dim, features_dim)

            enc_layer = TransformerEncoderLayer(
                d_model=features_dim, nhead=n_heads, dropout=dropout, batch_first=True
            )
            self.transformer = TransformerEncoder(enc_layer, num_layers=n_layers)
            self.layer_norm = nn.LayerNorm(features_dim)

            # checkpoints reference this param explicitly
            self.positional_encoding = nn.Parameter(th.zeros(1, 1, features_dim))

        def forward(self, obs: th.Tensor) -> th.Tensor:
            # shape: (B, obs_dim)
            if obs.ndim > 2:
                obs = th.flatten(obs, start_dim=1)
            x = self.linear_embed(obs)       # (B, features_dim)
            x = x.unsqueeze(1)               # (B, 1, features_dim)  (seq len = 1)
            x = x + self.positional_encoding
            x = self.transformer(x)          # (B, 1, features_dim)
            x = self.layer_norm(x)
            return x.squeeze(1)              # (B, features_dim)

    # --- shared env kwargs ---
    def _fetch_env_kwargs():
        g = globals()
        if "ENV_KWARGS" in g and g["ENV_KWARGS"] is not None:
            return dict(g["ENV_KWARGS"])
        needed = {
            "num_stock_shares", "buy_cost_list", "sell_cost_list",
            "state_space", "stock_dimension", "INDICATORS"
        }
        missing = [k for k in needed if k not in g]
        if missing:
            raise RuntimeError(
                "Env kwargs not configured. Either call set_env_kwargs({...}) once, "
                f"or define these globals before importing trading_dispatch: {missing}"
            )
        return {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "num_stock_shares": g["num_stock_shares"],
            "buy_cost_pct": g["buy_cost_list"],
            "sell_cost_pct": g["sell_cost_list"],
            "state_space": g["state_space"],
            "stock_dim": g["stock_dimension"],
            "tech_indicator_list": g["INDICATORS"],
            "action_space": g["stock_dimension"],
            "reward_scaling": 1e-4,
        }

    env_kwargs = _fetch_env_kwargs()

    # --- env factory (passes risk args if supported) ---
    def _make_env():
        try:
            env = StockTradingEnv(
                df=trade_df,
                turbulence_threshold=turbulence_threshold,
                risk_indicator_col=risk_indicator_col,
                **env_kwargs
            )
        except TypeError:
            env = StockTradingEnv(df=trade_df, **env_kwargs)
        # guard against FinRL's modulo-by-zero print bug
        try:
            if getattr(env, "print_verbosity", 1) == 0:
                env.print_verbosity = 10
        except Exception:
            pass
        return env

    def _portfolio_value_from_env(e):
        for k in ("portfolio_value", "account_value", "total_asset", "nav"):
            if hasattr(e, k):
                try:
                    return float(getattr(e, k))
                except Exception:
                    pass
        s = e.state
        sd = e.stock_dim
        cash = float(s[0])
        prices = np.array(s[1:1+sd], dtype=float)
        shares = np.array(s[1+sd:1+2*sd], dtype=float)
        return float(cash + (prices * shares).sum())

    def _run_one(model_file):
        if not os.path.isfile(model_file):
            return None

        env = _make_env()

        # --- Recreate the same policy arch as training so weights load cleanly ---
        policy_kwargs = dict(
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=32, n_heads=4, n_layers=1, dropout=0.1),
            share_features_extractor=False,   # matches separate pi_/vf_ extractors in your checkpoint
        )

        # Load with custom_objects so SB3 rebuilds compatible policy
        model = PPO.load(
            model_file,
            env=env,                 # bind env at load (ok with custom_objects)
            device="auto",
            custom_objects={"policy_kwargs": policy_kwargs},
        )

        out = env.reset()
        obs = out[0] if isinstance(out, tuple) and len(out) == 2 else out

        values = []
        prev_date = None

        while True:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if len(step_out) == 5:  # gymnasium
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:                   # gym
                obs, reward, done, info = step_out

            total_asset = _portfolio_value_from_env(env)
            cur_date = env.date_memory[-1] if hasattr(env, "date_memory") and env.date_memory else None

            # auto-reset guard: if date goes backwards, stop before logging
            if prev_date is not None and cur_date is not None and cur_date < prev_date:
                break

            values.append(total_asset)
            prev_date = cur_date

            if done:
                break

        if values:
            arr = np.asarray(values, dtype=float)
            rets = np.diff(arr) / (arr[:-1] + 1e-8)
            sharpe = (np.sqrt(252) * np.nanmean(rets) / (np.nanstd(rets) + 1e-8)) if len(rets) > 1 else 0.0
            print(f"[TransformerPPO] {os.path.basename(model_file)} — Final: {arr[-1]:.2f} | Sharpe: {sharpe:.4f}")
        return values if values else None

    # --- find models & run ---
    curves = []
    for i in range(loop_count):
        candidates = [os.path.join(model_path_base, f"TransformerPPO_5k_{i}.zip")]
        if not any(os.path.isfile(c) for c in candidates):
            globs = [
                os.path.join(model_path_base, f"*Transformer*PPO*{i}*.zip"),
                os.path.join(model_path_base, f"*Trans*PPO*{i}*.zip"),
                os.path.join(model_path_base, f"*transformerppo*{i}*.zip"),
            ]
            matches = []
            for gpat in globs:
                matches.extend(glob.glob(gpat))
            if not matches and i == 0:
                matches = glob.glob(os.path.join(model_path_base, "*transformer*ppo*.zip"))
            if matches:
                candidates = [sorted(matches)[0]]

        chosen = next((c for c in candidates if os.path.isfile(c)), None)
        if chosen is None:
            print(f"[TransformerPPO] Skip run {i}: no model file in {model_path_base}")
            continue

        vals = _run_one(chosen)
        if vals is not None and np.isfinite(np.asarray(vals)).any():
            curves.append(vals)
        else:
            print(f"[TransformerPPO] Run {i}: empty/NaN curve")

    return curves


def trade_moppo(
    model_path_base,
    trade_df,
    loop_count=5,
    turbulence_threshold=55.0,
    risk_indicator_col="vix",
):
    """
    Multi-Objective PPO trading runner for the dispatcher.
    Returns: List[List[float]] — one equity curve per loaded model.
    """
    import os, glob, json, numpy as np

    # --- prefer your local env; fallback to FinRL's ---
    try:
        from env_stocktrading import StockTradingEnv
    except Exception:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    # --- SB3 PPO ---
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise ImportError("PPO not found. Install stable-baselines3.") from e

    # --- shared env kwargs ---
    def _fetch_env_kwargs():
        g = globals()
        if "ENV_KWARGS" in g and g["ENV_KWARGS"] is not None:
            return dict(g["ENV_KWARGS"])
        needed = {
            "num_stock_shares", "buy_cost_list", "sell_cost_list",
            "state_space", "stock_dimension", "INDICATORS"
        }
        missing = [k for k in needed if k not in g]
        if missing:
            raise RuntimeError(
                "Env kwargs not configured. Either call set_env_kwargs({...}) once, "
                f"or define these globals before importing trading_dispatch: {missing}"
            )
        return {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "num_stock_shares": g["num_stock_shares"],
            "buy_cost_pct": g["buy_cost_list"],
            "sell_cost_pct": g["sell_cost_list"],
            "state_space": g["state_space"],
            "stock_dim": g["stock_dimension"],
            "tech_indicator_list": g["INDICATORS"],
            "action_space": g["stock_dimension"],
            "reward_scaling": 1e-4,
        }

    env_kwargs = _fetch_env_kwargs()

    # --- Multi-Objective wrapper (return − drawdown − cost) ---
    # Preserves Gym vs Gymnasium step API shape
    class _MOEnv:
        def __init__(self, base_env, w_return=1.0, w_drawdown=0.5, w_cost=0.1):
            self.base = base_env
            self.w_return = float(w_return)
            self.w_drawdown = float(w_drawdown)
            self.w_cost = float(w_cost)
            self.prev_action = None
            self.max_account_value = None

        def __getattr__(self, item):
            return getattr(self.base, item)

        def reset(self, **kwargs):
            out = self.base.reset(**kwargs)
            self.prev_action = None
            self.max_account_value = None
            # keep step API compatibility
            return out

        def step(self, action):
            step_out = self.base.step(action)
            # Unpack per API
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
                gymnasium_api = True
            else:
                obs, reward, done, info = step_out
                gymnasium_api = False

            reward_return = reward

            # Transaction cost proxy: L1 change of action vector
            try:
                cur_a = np.asarray(action, dtype=float).reshape(-1)
                prev_a = np.asarray(self.prev_action, dtype=float).reshape(-1) if self.prev_action is not None else None
                cost = float(np.sum(np.abs(cur_a - prev_a)) * 0.001) if prev_a is not None else 0.0
            except Exception:
                cost = 0.0
            self.prev_action = action

            # Drawdown proxy from account/asset memory (prefer latest portfolio value)
            try:
                if hasattr(self.base, "asset_memory") and self.base.asset_memory:
                    current_value = float(self.base.asset_memory[-1])
                else:
                    # compute from state if needed
                    s = self.base.state
                    sd = self.base.stock_dim
                    cash = float(s[0])
                    prices = np.array(s[1:1+sd], dtype=float)
                    shares = np.array(s[1+sd:1+2*sd], dtype=float)
                    current_value = float(cash + (prices * shares).sum())
                self.max_account_value = (
                    current_value if self.max_account_value is None else max(self.max_account_value, current_value)
                )
                drawdown = (self.max_account_value - current_value) / (self.max_account_value + 1e-10)
            except Exception:
                drawdown = 0.0

            shaped = (self.w_return * reward_return) - (self.w_cost * cost) - (self.w_drawdown * drawdown)

            if gymnasium_api:
                return obs, shaped, bool(terminated), bool(truncated), info
            else:
                return obs, shaped, done, info

        # proxy any missing attributes to base env
        def __getattr__(self, item):
            return getattr(self.base, item)

    # --- build env (wrap with MO weights) ---
    def _make_env(weights):
        try:
            base = StockTradingEnv(
                df=trade_df,
                turbulence_threshold=turbulence_threshold,
                risk_indicator_col=risk_indicator_col,
                **env_kwargs
            )
        except TypeError:
            base = StockTradingEnv(df=trade_df, **env_kwargs)
        return _MOEnv(
            base_env=base,
            w_return=weights.get("w_return", 1.0),
            w_drawdown=weights.get("w_drawdown", 0.5),
            w_cost=weights.get("w_cost", 0.1),
        )

    def _portfolio_value_from_env(e):
        for k in ("portfolio_value", "account_value", "total_asset", "nav"):
            if hasattr(e, k):
                try:
                    return float(getattr(e, k))
                except Exception:
                    pass
        s = e.state
        sd = e.stock_dim
        cash = float(s[0])
        prices = np.array(s[1:1+sd], dtype=float)
        shares = np.array(s[1+sd:1+2*sd], dtype=float)
        return float(cash + (prices * shares).sum())

    # --- sidecar loader for weights json ---
    def _load_weights_for_model(model_file):
        # 1) same stem: *.zip -> *.weights.json
        base, _ = os.path.splitext(model_file)
        cand = base + ".weights.json"
        if os.path.isfile(cand):
            path = cand
        else:
            # 2) look for any weights json next to the model
            folder = os.path.dirname(model_file)
            stem = os.path.basename(base)
            globs = [
                os.path.join(folder, f"{stem}*.weights.json"),
                os.path.join(folder, f"*weights*.json"),
                os.path.join(folder, "*.weights.json"),
            ]
            matches = []
            for gpat in globs:
                matches.extend(glob.glob(gpat))
            path = sorted(matches)[0] if matches else None

        if path is None:
            print(f"[MOPPO] No weights JSON found for {os.path.basename(model_file)}. Using defaults.")
            return {"w_return": 1.0, "w_drawdown": 0.5, "w_cost": 0.1}

        try:
            with open(path, "r") as f:
                w = json.load(f)
            # sanitize keys / defaults
            return {
                "w_return": float(w.get("w_return", 1.0)),
                "w_drawdown": float(w.get("w_drawdown", 0.5)),
                "w_cost": float(w.get("w_cost", 0.1)),
            }
        except Exception as e:
            print(f"[MOPPO] Failed to read weights from {os.path.basename(path)}: {e}. Using defaults.")
            return {"w_return": 1.0, "w_drawdown": 0.5, "w_cost": 0.1}

    def _run_one(model_file):
        if not os.path.isfile(model_file):
            return None

        weights = _load_weights_for_model(model_file)
        env = _make_env(weights)

        try:
            model = PPO.load(model_file, env=env, device="auto")
        except Exception:
            model = PPO.load(model_file, device="auto")

        out = env.reset()
        obs = out[0] if isinstance(out, tuple) and len(out) == 2 else out

        values = []
        prev_date = None

        while True:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if len(step_out) == 5:  # gymnasium
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:                   # gym
                obs, reward, done, info = step_out

            total_asset = _portfolio_value_from_env(env)
            cur_date = env.date_memory[-1] if hasattr(env, "date_memory") and env.date_memory else None

            # auto-reset guard
            if prev_date is not None and cur_date is not None and cur_date < prev_date:
                break

            values.append(total_asset)
            prev_date = cur_date

            if done:
                break

        if values:
            arr = np.asarray(values, dtype=float)
            rets = np.diff(arr) / (arr[:-1] + 1e-8)
            sharpe = (np.sqrt(252) * np.nanmean(rets) / (np.nanstd(rets) + 1e-8)) if len(rets) > 1 else 0.0
            print(
                f"[MOPPO] {os.path.basename(model_file)} — "
                f"Final: {arr[-1]:.2f} | Sharpe: {sharpe:.4f} | "
                f"w: R{weights['w_return']}, D{weights['w_drawdown']}, C{weights['w_cost']}"
            )
        return values if values else None

    # --- find models & run ---
    curves = []
    for i in range(loop_count):
        candidates = [os.path.join(model_path_base, f"MOPPO_5k_{i}.zip")]
        if not any(os.path.isfile(c) for c in candidates):
            globs = [
                os.path.join(model_path_base, f"*MO*PPO*{i}*.zip"),
                os.path.join(model_path_base, f"*Multi*Objective*PPO*{i}*.zip"),
                os.path.join(model_path_base, f"*moppo*{i}*.zip"),
            ]
            matches = []
            for gpat in globs:
                matches.extend(glob.glob(gpat))
            if not matches and i == 0:
                matches = glob.glob(os.path.join(model_path_base, "*mo*ppo*.zip"))
            if matches:
                candidates = [sorted(matches)[0]]

        chosen = next((c for c in candidates if os.path.isfile(c)), None)
        if chosen is None:
            print(f"[MOPPO] Skip run {i}: no model file in {model_path_base}")
            continue

        vals = _run_one(chosen)
        if vals is not None and np.isfinite(np.asarray(vals)).any():
            curves.append(vals)
        else:
            print(f"[MOPPO] Run {i}: empty/NaN curve")

    return curves

def trade_Baselineppo(
    model_path_base,
    trade_df,
    loop_count=5,
    turbulence_threshold=55.0,
    risk_indicator_col="vix",
):
    """
    Baseline PPO trading runner for the dispatcher.
    Returns: List[List[float]] — one equity curve per loaded model.
    """
    import os, glob, numpy as np

    # --- prefer your local env; fallback to FinRL's ---
    try:
        from env_stocktrading import StockTradingEnv
    except Exception:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    # --- SB3 PPO ---
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise ImportError("PPO not found. Install stable-baselines3.") from e

    # --- DRLAgent (prefer your models.py; else FinRL) ---
    DRLAgent = None
    try:
        from models import DRLAgent as _DRLAgent  # your project
        DRLAgent = _DRLAgent
    except Exception:
        try:
            from finrl.agents.stablebaselines3.models import DRLAgent as _DRLAgent  # FinRL
            DRLAgent = _DRLAgent
        except Exception:
            DRLAgent = None  # will fall back to manual loop

    # --- pull shared env kwargs (from set_env_kwargs) or infer from globals ---
    def _fetch_env_kwargs():
        g = globals()
        if "ENV_KWARGS" in g and g["ENV_KWARGS"] is not None:
            return dict(g["ENV_KWARGS"])
        needed = {
            "num_stock_shares", "buy_cost_list", "sell_cost_list",
            "state_space", "stock_dimension", "INDICATORS"
        }
        missing = [k for k in needed if k not in g]
        if missing:
            raise RuntimeError(
                "Env kwargs not configured. Either call set_env_kwargs({...}) once, "
                f"or define these globals before importing trading_dispatch: {missing}"
            )
        return {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "num_stock_shares": g["num_stock_shares"],
            "buy_cost_pct": g["buy_cost_list"],
            "sell_cost_pct": g["sell_cost_list"],
            "state_space": g["state_space"],
            "stock_dim": g["stock_dimension"],
            "tech_indicator_list": g["INDICATORS"],
            "action_space": g["stock_dimension"],
            "reward_scaling": 1e-4,
        }

    env_kwargs = _fetch_env_kwargs()

    # --- env factory (passes risk args if supported) ---
    def _make_env():
        try:
            return StockTradingEnv(
                df=trade_df,
                turbulence_threshold=turbulence_threshold,
                risk_indicator_col=risk_indicator_col,
                **env_kwargs
            )
        except TypeError:
            return StockTradingEnv(df=trade_df, **env_kwargs)

    def _portfolio_value_from_env(e):
        for k in ("portfolio_value", "account_value", "total_asset", "nav"):
            if hasattr(e, k):
                try:
                    return float(getattr(e, k))
                except Exception:
                    pass
        s = e.state
        sd = e.stock_dim
        cash = float(s[0])
        prices = np.array(s[1:1+sd], dtype=float)
        shares = np.array(s[1+sd:1+2*sd], dtype=float)
        return float(cash + (prices * shares).sum())

    def _run_one(model_file):
        if not os.path.isfile(model_file):
            return None

        env = _make_env()

        # Load model (bind env if possible)
        try:
            model = PPO.load(model_file, env=env, device="auto")
        except Exception:
            model = PPO.load(model_file, device="auto")

        # Preferred: DRLAgent pathway (consistent with your snippet)
        if DRLAgent is not None:
            try:
                df_account_value, _ = DRLAgent.DRL_prediction(model=model, environment=env)
                vals = df_account_value["account_value"].tolist()
                if vals:
                    arr = np.asarray(vals, dtype=float)
                    rets = np.diff(arr) / (arr[:-1] + 1e-8)
                    sharpe = (np.sqrt(252) * np.nanmean(rets) / (np.nanstd(rets) + 1e-8)) if len(rets) > 1 else 0.0
                    print(f"[PPO] {os.path.basename(model_file)} — Final: {arr[-1]:.2f} | Sharpe: {sharpe:.4f}")
                return vals if vals else None
            except Exception:
                # Fall back to manual loop if DRLAgent path fails
                pass

        # Manual loop (Gym/Gymnasium safe)
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) and len(out) == 2 else out

        values = []
        prev_date = None
        while True:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if len(step_out) == 5:  # gymnasium
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:                   # gym
                obs, reward, done, info = step_out

            total_asset = _portfolio_value_from_env(env)
            cur_date = env.date_memory[-1] if hasattr(env, "date_memory") and env.date_memory else None

            # auto-reset guard: if date goes backwards, break before logging
            if prev_date is not None and cur_date is not None and cur_date < prev_date:
                break

            values.append(total_asset)
            prev_date = cur_date

            if done:
                break

        if values:
            arr = np.asarray(values, dtype=float)
            rets = np.diff(arr) / (arr[:-1] + 1e-8)
            sharpe = (np.sqrt(252) * np.nanmean(rets) / (np.nanstd(rets) + 1e-8)) if len(rets) > 1 else 0.0
            print(f"[PPO] {os.path.basename(model_file)} — Final: {arr[-1]:.2f} | Sharpe: {sharpe:.4f}")
        return values if values else None

    # --- find models & run ---
    curves = []
    for i in range(loop_count):
        # Your original pattern first
        candidates = [os.path.join(model_path_base, f"ppo_model_run{i}.zip")]
        # Additional common patterns (flexible)
        if not any(os.path.isfile(c) for c in candidates):
            candidates.append(os.path.join(model_path_base, f"PPO_5k_{i}.zip"))
        if not any(os.path.isfile(c) for c in candidates):
            globs = [
                os.path.join(model_path_base, f"*ppo*run*{i}*.zip"),
                os.path.join(model_path_base, f"*ppo*{i}*.zip"),
                os.path.join(model_path_base, f"*PPO*{i}*.zip"),
            ]
            matches = []
            for gpat in globs:
                matches.extend(glob.glob(gpat))
            if matches:
                candidates = [sorted(matches)[0]]

        chosen = next((c for c in candidates if os.path.isfile(c)), None)
        if chosen is None:
            print(f"[PPO] Skip run {i}: no model file in {model_path_base}")
            continue

        vals = _run_one(chosen)
        if vals is not None and np.isfinite(np.asarray(vals)).any():
            curves.append(vals)
        else:
            print(f"[PPO] Run {i}: empty/NaN curve")

    return curves
