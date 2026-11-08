"""Measure raw env steps/sec: envpool vs gymnasium vector envs (random actions)."""
import time, sys, json, numpy as np
res = {}
def bench_envpool(n, threads):
    import envpool
    env = envpool.make("Breakout-v5", env_type="gym", num_envs=n, batch_size=n, num_threads=threads,
                       episodic_life=True, reward_clip=True, repeat_action_probability=0.0,
                       noop_max=30, full_action_space=False, max_episode_steps=int(108000/4))
    env.reset(); a = np.zeros(n, dtype=np.int32)
    t = time.time(); steps = 0
    while time.time() - t < 8:
        env.step(np.random.randint(0, 4, n).astype(np.int32)); steps += n
    return steps / (time.time() - t)
def bench_gym(n):
    import gymnasium as gym, ale_py
    gym.register_envs(ale_py)
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
    def mk():
        e = gym.make("BreakoutNoFrameskip-v4"); e = AtariPreprocessing(e, frame_skip=4, noop_max=30, terminal_on_life_loss=True)
        return FrameStackObservation(e, 4)
    env = gym.vector.AsyncVectorEnv([mk]*n); env.reset(seed=0)
    t = time.time(); steps = 0
    while time.time() - t < 8:
        env.step(np.random.randint(0, 4, n)); steps += n
    env.close(); return steps / (time.time() - t)
for n, th in [(8,8),(16,8),(32,8)]:
    res[f"envpool_n{n}"] = bench_envpool(n, th); print(f"envpool n={n}: {res[f'envpool_n{n}']:.0f} agent-steps/s", flush=True)
for n in [8,16]:
    res[f"gym_async_n{n}"] = bench_gym(n); print(f"gym async n={n}: {res[f'gym_async_n{n}']:.0f}", flush=True)
json.dump(res, open("bench/env_bench.json","w"), indent=1)
