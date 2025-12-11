import os
import csv
import argparse
from datetime import datetime
import gymnasium as gym 
import panda_gym
import numpy as np
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from typing import Optional, Dict, Any, List
   

class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, gravity_mean=9.8, gravity_std=0.0, rest_mean=0.5, rest_std=0.0,
                 fric_mean=0.5, fric_lat_std=0.0, fric_roll_std=0.0, fric_spin_std=0.0,
                 mass_mean=1.0, mass_std=0.0, mass_min=1e-3, mass_max=1e3,
                 include_bodies=("object", "cube", "box", "table", "floor", "ground"),
                 exclude_bodies=("panda", "franka"), include_links=(), exclude_links=(),
                 log_targets_once=True, dist="normal"):
        super().__init__(env)
        self.params = dict(
            gravity_mean=gravity_mean, gravity_std=gravity_std,
            rest_mean=rest_mean, rest_std=rest_std,
            fric_mean=fric_mean,
            fric_lat_std=fric_lat_std, fric_roll_std=fric_roll_std, fric_spin_std=fric_spin_std,
            mass_mean=mass_mean, mass_std=mass_std,
            mass_min=mass_min, mass_max=mass_max,
        )
        self.include_bodies = tuple(s.lower() for s in include_bodies or ())
        self.exclude_bodies = tuple(s.lower() for s in exclude_bodies or ())
        self.include_links = tuple(s.lower() for s in include_links or ())
        self.exclude_links = tuple(s.lower() for s in exclude_links or ())
        self._logged_targets = not log_targets_once
        assert dist in ("normal", "uniform")
        self.dist = dist
        self._current_dr_params: Dict[str, float] = {}

    def _get_bullet_client(self):
        cand = []
        un = self.env.unwrapped
        for attr_path in [
            ("sim", "physics_client"),
            ("physics_client",),
            ("_p",),
            ("env", "sim", "physics_client"),
        ]:
            obj = un
            ok = True
            for a in attr_path:
                if hasattr(obj, a):
                    obj = getattr(obj, a)
                else:
                    ok = False
                    break
            if ok:
                cand.append(obj)
        if not cand:
            raise RuntimeError("PyBullet client not found.")
        return cand[0]

    def _sample_param(self, mean, std, min_val=None, max_val=None) -> float:
        if std <= 0:
            x = mean
        else:
            if self.dist == "uniform":
                low = mean - std
                high = mean + std
                if min_val is not None:
                    low = max(low, min_val)
                if max_val is not None:
                    high = min(high, max_val)
                if low > high:
                    low, high = high, low
                x = np.random.uniform(low, high)
            else:
                x = np.random.normal(mean, std)
        if min_val is not None:
            x = max(min_val, x)
        if max_val is not None:
            x = min(max_val, x)
        return float(x)

    def _iter_target_links(self, client):
        def _match(name_bytes, includes, excludes):
            name = (name_bytes or b"").decode("utf-8", errors="ignore").lower()
            if excludes and any(x in name for x in excludes):
                return False
            if includes:
                return any(x in name for x in includes)
            return True

        targets = []
        n_bodies = client.getNumBodies()
        for bi in range(n_bodies):
            bid = client.getBodyUniqueId(bi)
            try:
                body_name = client.getBodyInfo(bid)[1]
            except Exception:
                body_name = b""

            if not _match(body_name, self.include_bodies, self.exclude_bodies):
                continue

            try:
                n_joints = client.getNumJoints(bid)
            except Exception:
                n_joints = 0

            if not self.include_links:
                targets.append((bid, -1, b"base"))

            for ji in range(n_joints):
                try:
                    link_name = client.getJointInfo(bid, ji)[12]
                except Exception:
                    link_name = b""
                if _match(link_name, self.include_links, self.exclude_links):
                    targets.append((bid, ji, link_name))

        if not self._logged_targets:
            for bid, li, lname in targets:
                try:
                    bname = client.getBodyInfo(bid)[1].decode("utf-8", "ignore")
                except Exception:
                    bname = f"body_{bid}"
                print(f"[DR] body='{bname}', link_index={li}")
            self._logged_targets = True

        return targets

    def _apply_randomization(self):
        client = self._get_bullet_client()
        g = self._sample_param(self.params["gravity_mean"], self.params["gravity_std"], min_val=0.0)
        client.setGravity(0, 0, -g)
        restitution = self._sample_param(self.params["rest_mean"], self.params["rest_std"], min_val=0.0)
        fric_lat = self._sample_param(self.params["fric_mean"], self.params["fric_lat_std"], min_val=0.0)
        fric_roll = self._sample_param(self.params["fric_mean"], self.params["fric_roll_std"], min_val=0.0)
        fric_spin = self._sample_param(self.params["fric_mean"], self.params["fric_spin_std"], min_val=0.0)
        mass_val = self._sample_param(
            self.params["mass_mean"], self.params["mass_std"],
            min_val=self.params["mass_min"], max_val=self.params["mass_max"]
        )
        self._current_dr_params = dict(
            gravity=g, restitution=restitution,
            fric_lateral=fric_lat, fric_rolling=fric_roll,
            fric_spinning=fric_spin, mass=mass_val
        )

        for body_id, link_index, _ in self._iter_target_links(client):
            kwargs = dict(
                restitution=restitution,
                lateralFriction=fric_lat,
                rollingFriction=fric_roll,
                spinningFriction=fric_spin,
                mass=mass_val,
            )
            try:
                client.changeDynamics(body_id, link_index, **kwargs)
            except TypeError:
                for k, v in kwargs.items():
                    try:
                        client.changeDynamics(body_id, link_index, **{k: v})
                    except Exception:
                        pass

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_randomization()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            if isinstance(info, dict):
                info = info.copy()
            info["dr_params"] = self._current_dr_params.copy()
        return obs, reward, terminated, truncated, info

def make_env(env_id: str, seed: int, idx: int, reward_type: str = "sparse",
             dr_args: Optional[Dict[str, Any]] = None):
    def _init():
        env = gym.make(env_id, reward_type=reward_type)
        env = Monitor(env)
        env.reset(seed=seed + idx)
        if dr_args is not None:
            env = DomainRandomizationWrapper(env, **dr_args)
        return env
    return _init

class DRLoggingCallback(BaseCallback):
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "env_idx",
                    "gravity", "restitution",
                    "fric_lateral", "fric_rolling", "fric_spinning",
                    "mass", "is_success", "episode_reward", "episode_length",
                ])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for env_idx, (info, done) in enumerate(zip(infos, dones)):
            if not done:
                continue
            if "dr_params" not in info:
                continue

            params = info["dr_params"]
            is_success = info.get("is_success", None)
            ep_info = info.get("episode", {})
            ep_rew = ep_info.get("r", None)
            ep_len = ep_info.get("l", None)

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps, env_idx,
                    params.get("gravity"),
                    params.get("restitution"),
                    params.get("fric_lateral"),
                    params.get("fric_rolling"),
                    params.get("fric_spinning"),
                    params.get("mass"),
                    is_success, ep_rew, ep_len
                ])
        return True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="PandaPush-v3")
    parser.add_argument("--algo", type=str, default="td3", choices=["td3", "sac"])
    parser.add_argument("--reward-type", type=str, default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=250_000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr-sac", type=float, default=3e-4)
    parser.add_argument("--lr-td3", type=float, default=1e-3)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument("--deterministic-eval", action="store_true", default=True)
    parser.add_argument("--train-freq", type=int, default=10)
    parser.add_argument("--gradient-steps", type=int, default=10)
    parser.add_argument("--gravity-mean", type=float, default=9.8)
    parser.add_argument("--gravity-std", type=float, default=0.1)
    parser.add_argument("--rest-mean", type=float, default=0.5)
    parser.add_argument("--rest-std", type=float, default=0.1)
    parser.add_argument("--fric-mean", type=float, default=0.5)
    parser.add_argument("--fric-std", type=float, default=0.2)
    parser.add_argument("--mass-mean", type=float, default=1.0)
    parser.add_argument("--mass-std", type=float, default=0.2)
    parser.add_argument("--mass-min", type=float, default=1e-3)
    parser.add_argument("--mass-max", type=float, default=1e3)
    parser.add_argument("--dr-mode", type=str, default="seq5",
                        choices=["seq5", "all5", "sweep_seq5", "sweep_all5"])
    parser.add_argument("--std-scale-list", type=str, default="")
    parser.add_argument("--std-scale-start", type=float, default=0.5)
    parser.add_argument("--std-scale-end", type=float, default=1.5)
    parser.add_argument("--std-scale-steps", type=int, default=5)
    parser.add_argument("--dr-dist", type=str, default="normal", choices=["normal", "uniform"])
    return parser.parse_args()

def build_dr_args_for(args, target: str, scale: float = 1.0) -> Dict[str, Any]:
    fric_lat_std = (args.fric_std * scale) if target == "fric_lateral" else 0.0
    fric_roll_std = (args.fric_std * scale) if target == "fric_rolling" else 0.0
    fric_spin_std = 0.0
    return dict(
        gravity_mean=args.gravity_mean,
        gravity_std=(args.gravity_std * scale) if target == "gravity" else 0.0,
        rest_mean=args.rest_mean,
        rest_std=(args.rest_std * scale) if target == "restitution" else 0.0,
        fric_mean=args.fric_mean,
        fric_lat_std=fric_lat_std,
        fric_roll_std=fric_roll_std,
        fric_spin_std=fric_spin_std,
        mass_mean=args.mass_mean,
        mass_std=(args.mass_std * scale) if target == "mass" else 0.0,
        mass_min=args.mass_min,
        mass_max=args.mass_max,
        include_bodies=("object", "cube", "box", "table", "floor", "ground"),
        exclude_bodies=("panda", "franka"),
        dist=args.dr_dist,
    )

def build_all5_args(args, scale: float = 1.0) -> Dict[str, Any]:
    return dict(
        gravity_mean=args.gravity_mean, gravity_std=args.gravity_std * scale,
        rest_mean=args.rest_mean, rest_std=args.rest_std * scale,
        fric_mean=args.fric_mean,
        fric_lat_std=args.fric_std * scale,
        fric_roll_std=args.fric_std * scale,
        fric_spin_std=0.0,
        mass_mean=args.mass_mean, mass_std=args.mass_std * scale,
        mass_min=args.mass_min, mass_max=args.mass_max,
        include_bodies=("object", "cube", "box", "table", "floor", "ground"),
        exclude_bodies=("panda", "franka"),
        dist=args.dr_dist,
    )

def parse_scale_list(args) -> List[float]:
    if args.std_scale_list.strip():
        return [float(x) for x in args.std_scale_list.split(",")]
    if args.std_scale_steps <= 1:
        return [args.std_scale_start]
    return list(np.linspace(args.std_scale_start, args.std_scale_end, args.std_scale_steps))

def train_once(args, dr_args, run_suffix: str):
    run_name = f"{args.env_id}-{args.algo}-{args.reward_type}-{run_suffix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tb_logdir = os.path.join(args.logdir, run_name)
    os.makedirs(tb_logdir, exist_ok=True)

    env_fns = [make_env(args.env_id, args.seed, i, reward_type=args.reward_type, dr_args=dr_args)
               for i in range(args.n_envs)]
    train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([
        make_env(args.env_id, args.seed + 10000, 0, reward_type=args.reward_type, dr_args=dr_args)
    ])
    eval_env = VecMonitor(eval_env)

    her_kwargs = dict(
        n_sampled_goal=int(args.n_sampled_goal),
        goal_selection_strategy="future",
    )
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

    action_dim = int(np.prod(train_env.action_space.shape))
    if args.algo == "td3":
        algo_cls = TD3
        lr = args.lr_td3
        action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))
    else:
        algo_cls = SAC
        lr = args.lr_sac
        action_noise = None

    model = algo_cls(
        policy="MultiInputPolicy",
        env=train_env,
        tensorboard_log=tb_logdir,
        buffer_size=int(args.buffer_size),
        learning_rate=lr,
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        tau=float(args.tau),
        train_freq=(int(args.train_freq), "step"),
        gradient_steps=int(args.gradient_steps),
        learning_starts=10000,
        policy_kwargs=policy_kwargs,
        action_noise=action_noise,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=her_kwargs,
        verbose=1,
        seed=args.seed,
    )

    ckpt_dir = os.path.join(tb_logdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    eval_freq_vec = max(1, args.eval_freq // max(1, args.n_envs))
    save_freq_vec = max(1, args.save_freq // max(1, args.n_envs))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(tb_logdir, "best_model"),
        log_path=os.path.join(tb_logdir, "eval"),
        eval_freq=eval_freq_vec,
        n_eval_episodes=args.eval_episodes,
        deterministic=args.deterministic_eval,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_vec,
        save_path=ckpt_dir,
        name_prefix="panda_push",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    dr_log_path = os.path.join(tb_logdir, "dr_episode_log.csv")
    dr_logging_callback = DRLoggingCallback(log_path=dr_log_path)

    model.learn(
        total_timesteps=int(args.total_steps),
        progress_bar=True,
        callback=[eval_callback, checkpoint_callback, dr_logging_callback],
        log_interval=10,
    )

    model.save(os.path.join(tb_logdir, "final_model"))
    print(f"[Done:{run_suffix}] Logs: {tb_logdir}")
    print(f"tensorboard --logdir {args.logdir}")
    print(f"DR episode log saved to: {dr_log_path}")

def run_seq5(args, scale: float = 1.0):
    targets = ["gravity", "restitution", "fric_lateral", "fric_rolling", "mass"]
    for t in targets:
        dr_args_t = build_dr_args_for(args, t, scale=scale)
        train_once(args, dr_args_t, run_suffix=f"{t.upper()}-SCALE{scale:.2f}")

def run_all5(args, scale: float = 1.0):
    dr_args_all = build_all5_args(args, scale=scale)
    train_once(args, dr_args_all, run_suffix=f"ALL5-SCALE{scale:.2f}")

def main():
    args = parse_args()

    if args.dr_mode in ("seq5", "all5"):
        if args.dr_mode == "seq5":
            run_seq5(args, scale=1.0)
        else:
            run_all5(args, scale=1.0)
        return

    scales = parse_scale_list(args)
    print(f"[SWEEP] std scales = {scales} (dist={args.dr_dist})")

    if args.dr_mode == "sweep_seq5":
        for s in scales:
            run_seq5(args, scale=s)
    elif args.dr_mode == "sweep_all5":
        for s in scales:
            run_all5(args, scale=s)
    else:
        raise ValueError(f"Unknown dr-mode: {args.dr_mode}")

if __name__ == "__main__":
    main()
