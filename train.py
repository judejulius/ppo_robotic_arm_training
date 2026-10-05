import tensorflow as tf
import numpy as np

import argparse
import json
import math
import platform
import time
from collections import deque
from functools import partial
from pathlib import Path

import gymnasium as gym
from runtime import make_env
import importlib.metadata

from tensor_optix.algorithms.tf_ppo_continuous import TFGaussianPPOAgent
from tensor_optix.core.base_pipeline import BasePipeline
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.loop_controller import LoopCallback
from tensor_optix.core.types import EpisodeData, EvalMetrics, HyperparamSet
from tensor_optix.optimizer import RLOptimizer as Optimizer

HP = dict(learning_rate=3e-4, clip_ratio=0.2, entropy_coef=0.001,
          vf_coef=0.5, gamma=0.99, gae_lambda=0.95, n_epochs=10,
          minibatch_size=64, max_grad_norm=0.5)
ENV = 'FetchPushDense-v4'
VALIDATION_SEEDS = list(range(100000, 100020))
FINAL_SEEDS = list(range(400000, 400100))
DISPLAY_SEEDS = list(range(300000, 300005))
MAX_BUDGET = 10_000_384




def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


class Moments:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        delta = x.mean(axis=0) - self.mean
        count = self.count + len(x)
        self.var = (self.var*self.count + x.var(axis=0)*len(x)
                    + delta**2*self.count*len(x)/count) / count
        self.mean += delta*len(x)/count
        self.count = count

    def normalize(self, x):
        return np.clip((x-self.mean)/np.sqrt(self.var+1e-8), -10, 10).astype(np.float32)

    def state(self):
        return dict(mean=self.mean.tolist(), var=self.var.tolist(), count=self.count)

    def restore(self, state):
        self.mean = np.asarray(state['mean'], np.float64)
        self.var = np.asarray(state['var'], np.float64)
        self.count = state['count']


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Time-major [steps, environments]; never join separate environments."""
    delta = rewards + gamma*next_values*(1-terminated) - values
    out = np.empty_like(values, dtype=np.float32)
    carry = np.zeros(values.shape[1], np.float32)
    for t in reversed(range(len(values))):
        carry = delta[t] + gamma*lam*(1-np.maximum(terminated[t], truncated[t]))*carry
        out[t] = carry
    return out, out + values


def network(outputs, name, final_gain):
    inputs = tf.keras.Input((28,), name='normalized_state')
    x = inputs
    for i in range(2):
        x = tf.keras.layers.Dense(64, activation='tanh',
            kernel_initializer=tf.keras.initializers.Orthogonal(math.sqrt(2)),
            name=f'hidden_{i+1}')(x)
    x = tf.keras.layers.Dense(outputs, kernel_initializer=tf.keras.initializers.Orthogonal(final_gain),
                              name='output')(x)
    return tf.keras.Model(inputs, x, name=name)


class RobotAgent(TFGaussianPPOAgent):
    """Local Tensor Optix adapter: batched collection, correct GAE, compiled PPO.

    Keeps the installed agent's loss and distribution, retaining pre-tanh
    samples instead of inverting rounded actions. No installed code is changed.
    """
    def __init__(self, seed=42):
        super().__init__(network(6, 'actor', .01), network(1, 'critic', 1),
                         tf.keras.optimizers.Adam(HP['learning_rate']), 3,
                         HyperparamSet(dict(HP), 0))
        self._optimizer.build(self._actor.trainable_variables + self._critic.trainable_variables)
        self.obs_stats = Moments((28,))
        self.reward_stats = Moments()
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.updates = 0
        self.last_diagnostics = {}
        self.sync_numpy()

    def sync_numpy(self):
        # Small dense inference on CPU avoids thousands of GPU transfers.
        self.actor_arrays = self._actor.get_weights()
        self.critic_arrays = self._critic.get_weights()

    @staticmethod
    def dense(x, weights):
        for k in range(0, len(weights), 2):
            x = x @ weights[k] + weights[k+1]
            if k < len(weights)-2:
                x = np.tanh(x)
        return x

    def predict(self, normalized_obs, stochastic=False):
        out = self.dense(normalized_obs, self.actor_arrays)
        mean, log_std = out[:, :3], np.clip(out[:, 3:], -5, 2)
        u = mean + np.exp(log_std)*self.rng.normal(size=mean.shape).astype(np.float32) if stochastic else mean
        action = np.tanh(u)
        lp = np.sum(-.5*((u-mean)/(np.exp(log_std)+1e-8))**2-log_std
                    -.5*np.log(2*np.pi)-np.log(1-action**2+1e-6), axis=-1)
        value = self.dense(normalized_obs, self.critic_arrays)[:, 0]
        return action, u, lp, value

    @tf.function(reduce_retracing=True)
    def train_rollout(self, obs, latent, old_lp, advantages, returns):
        variables = self._actor.trainable_variables + self._critic.trainable_variables
        sums = tf.zeros(5)
        updates = tf.constant(0)
        stop = tf.constant(False)
        n = tf.shape(obs)[0]
        batches = n // HP['minibatch_size']
        for epoch in tf.range(HP['n_epochs']):
            if stop:
                break
            indices = tf.random.shuffle(tf.range(n))
            for i in tf.range(batches):
                idx = indices[i*64:(i+1)*64]
                x, u, old, adv, ret = [tf.gather(a, idx) for a in (obs, latent, old_lp, advantages, returns)]
                with tf.GradientTape() as tape:
                    out = self._actor(x, training=True)
                    mu, ls = out[:, :3], tf.clip_by_value(out[:, 3:], -5., 2.)
                    lp = self._gaussian_log_prob(u, mu, ls)
                    ratio = tf.exp(lp-old)
                    pol = -tf.reduce_mean(tf.minimum(ratio*adv, tf.clip_by_value(ratio, .8, 1.2)*adv))
                    val = tf.reduce_mean(tf.square(tf.squeeze(self._critic(x, training=True), -1)-ret))
                    ent = tf.reduce_mean(tf.reduce_sum(ls + .5*(1+np.log(2*np.pi)), axis=-1))
                    loss = pol + .5*val - .001*ent
                gradients = tape.gradient(loss, variables)
                gradients, _ = tf.clip_by_global_norm(gradients, .5)
                self._optimizer.apply_gradients(zip(gradients, variables))
                kl = tf.reduce_mean((ratio-1)-(lp-old))
                clip_fraction = tf.reduce_mean(tf.cast(tf.abs(ratio-1) > .2, tf.float32))
                tf.debugging.assert_all_finite(loss, 'Nonfinite PPO loss')
                sums += tf.stack([pol, val, ent, kl, clip_fraction])
                updates += 1
                # Check divergence across the whole rollout, not a noisy minibatch.
            full = self._actor(obs, training=False)
            new_lp = self._gaussian_log_prob(latent, full[:, :3], tf.clip_by_value(full[:, 3:], -5., 2.))
            log_ratio = new_lp-old_lp
            stop = tf.reduce_mean(tf.exp(log_ratio)-1-log_ratio) > .02
        return sums / tf.cast(updates, tf.float32)

    def learn(self, data):
        self._optimizer.learning_rate.assign(3e-4 - 2.7e-4*min(self.steps/MAX_BUDGET, 1.))
        m = data.metadata
        adv, returns = gae(m['rewards'], m['values'], m['next_values'],
                           m['terminated'], m['truncated'], HP['gamma'], HP['gae_lambda'])
        adv = (adv-adv.mean())/(adv.std()+1e-8)
        metrics = self.train_rollout(*[tf.convert_to_tensor(a, tf.float32) for a in
            (data.observations, m['latent'].reshape(-1, 3), m['log_probs'].ravel(), adv.ravel(), returns.ravel())]).numpy()
        if not np.all(np.isfinite(metrics)):
            raise FloatingPointError(f'Nonfinite PPO diagnostics: {metrics}')
        self.obs_stats.update(m['raw_observations'].reshape(-1, 28))
        self.sync_numpy()
        self.steps += data.length
        self.updates += 1
        self.last_diagnostics = dict(zip(['policy_loss', 'value_loss', 'pre_squash_entropy', 'approx_kl', 'clip_fraction'], map(float, metrics)))
        self.last_diagnostics['learning_rate'] = float(self._optimizer.learning_rate.numpy())
        self.last_diagnostics['explained_var'] = float(1-np.var(returns-m['values'])/(np.var(returns)+1e-8))
        return self.last_diagnostics

    def save_weights(self, path):
        super().save_weights(path)
        write_json(Path(path)/'state.json', dict(obs=self.obs_stats.state(), reward=self.reward_stats.state(),
            steps=self.steps, updates=self.updates, rng=self.rng.bit_generator.state))
        np.savez(Path(path)/'adam.npz', **{f'v{i}': v.numpy() for i, v in enumerate(self._optimizer.variables)})

    def load_weights(self, path):
        super().load_weights(path)
        state = json.loads((Path(path)/'state.json').read_text())
        self.obs_stats.restore(state['obs'])
        self.reward_stats.restore(state['reward'])
        self.steps, self.updates = state['steps'], state['updates']
        self.rng.bit_generator.state = state['rng']
        with np.load(Path(path)/'adam.npz') as arrays:
            for i, v in enumerate(self._optimizer.variables):
                v.assign(arrays[f'v{i}'])
        self.sync_numpy()


class RobotPipeline(BasePipeline):
    def __init__(self, agent, output, max_steps, n_envs=16, seed=42, approach_weight=0.):
        self.agent, self.output, self.max_steps = agent, output, max_steps
        self.n_envs, self.seed = n_envs, seed
        self.approach_weight = approach_weight
        self.collected = agent.steps
        self.completed = 0
        self.recent = deque(maxlen=100)
        self.halt = False

    def setup(self):
        self.envs = gym.vector.AsyncVectorEnv([partial(make_env, approach_weight=self.approach_weight)]*self.n_envs,
            context='fork', shared_memory=False, autoreset_mode=gym.vector.AutoresetMode.DISABLED)
        # Fresh seeded episodes on resume; optimizer/statistics resume exactly,
        # MuJoCo physics and TF minibatch shuffle do not claim bitwise continuation.
        self.obs, _ = self.envs.reset(seed=self.seed+10*self.agent.steps)
        self.returns = np.zeros(self.n_envs)
        self.native_returns = np.zeros(self.n_envs)
        self.lengths = np.zeros(self.n_envs, int)
        self.discounted = np.zeros(self.n_envs)
        self.episode_file = (self.output/'episodes.jsonl').open('a', buffering=1)

    def episodes(self):
        while self.collected < self.max_steps and not self.halt:
            horizon = min(2048, self.max_steps-self.collected)//self.n_envs
            if horizon*self.n_envs < 64:
                break
            fields = {k: [] for k in ['observations', 'raw_observations', 'actions', 'latent', 'log_probs',
                'values', 'next_values', 'rewards', 'terminated', 'truncated']}
            for _ in range(horizon):
                raw = self.obs.copy()
                x = self.agent.obs_stats.normalize(raw)
                actions, u, lp, values = self.agent.predict(x, stochastic=True)
                next_obs, rewards, term, trunc, _info = self.envs.step(actions)
                if not np.all(np.isfinite(next_obs)) or not np.all(np.isfinite(rewards)):
                    raise FloatingPointError('Nonfinite environment transition')
                self.returns += rewards
                self.native_returns += _info['native_reward']
                self.lengths += 1
                ended = term | trunc
                self.obs = next_obs.copy()
                for i in np.flatnonzero(ended):
                    record = dict(steps=self.collected+int(i)+1, reward=float(self.returns[i]),
                                  length=int(self.lengths[i]), terminated=bool(term[i]), truncated=bool(trunc[i]),
                                  native_reward=float(self.native_returns[i]),
                                  approach_distance_cm=float(_info['approach_distance_cm'][i]),
                                  block_displacement_cm=float(_info['block_displacement_cm'][i]))
                    self.episode_file.write(json.dumps(record)+'\n')
                    self.recent.append(float(self.returns[i]))
                    self.completed += 1
                    self.returns[i], self.lengths[i], self.native_returns[i] = 0, 0, 0
                if np.any(ended):
                    self.obs, _ = self.envs.reset(options={'reset_mask': ended.copy()})
                next_values = self.agent.dense(self.agent.obs_stats.normalize(np.array(next_obs)), self.agent.critic_arrays)[:, 0]
                self.discounted = HP['gamma']*self.discounted + rewards
                self.agent.reward_stats.update(self.discounted)
                scaled = np.clip(np.asarray(rewards)/np.sqrt(self.agent.reward_stats.var+1e-8), -10, 10)
                self.discounted[np.asarray(term)|np.asarray(trunc)] = 0
                entries = [x, raw, actions, u, lp, values, next_values, scaled, term, trunc]
                for key, value in zip(fields, entries):
                    fields[key].append(value)
                self.collected += self.n_envs
            m = {k: np.asarray(v, np.float32) for k, v in fields.items()}
            yield EpisodeData(observations=m['observations'].reshape(-1, 28),
                actions=m['actions'].reshape(-1, 3), rewards=m['rewards'].ravel().tolist(),
                terminated=m['terminated'].ravel().astype(bool).tolist(),
                truncated=m['truncated'].ravel().astype(bool).tolist(), infos=[], episode_id=self.agent.updates,
                metadata=m)

    def teardown(self):
        self.envs.close()
        self.episode_file.close()

    @property
    def is_live(self):
        return False


class FixedParameters(BaseOptimizer):
    def suggest(self, current_hyperparams, metrics_history):
        return current_hyperparams.copy()


class TrainingScore(BaseEvaluator):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def score(self, episode_data, train_diagnostics):
        reward = float(np.mean(self.pipeline.recent)) if self.pipeline.recent else -1000.
        return EvalMetrics(reward, train_diagnostics or {}, episode_data.episode_id)


def evaluate(agent, seeds, random=False):
    env = make_env()
    rows = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            total, length, any_success = 0., 0, False
            while True:
                action = env.action_space.sample() if random else agent.predict(agent.obs_stats.normalize(obs[None]))[0][0]
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                length += 1
                any_success = any_success or bool(info['is_success'])
                if terminated or truncated:
                    break
            rows.append(dict(seed=seed, reward=float(total), length=length,
                success=bool(info['is_success']), any_success=any_success,
                final_distance_cm=float(np.linalg.norm(obs[3:6]-obs[25:28])*100),
                approach_distance_cm=float(info['approach_distance_cm']),
                block_displacement_cm=float(info['block_displacement_cm'])))
    finally:
        env.close()
    scores = [r['reward'] for r in rows]
    n = len(rows)
    p = sum(r['success'] for r in rows)/n
    z = 1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(mean_reward=float(np.mean(scores)), std_reward=float(np.std(scores)),
        episodes=n, success_rate=p, success_95ci=[center-radius, center+radius],
        mean_final_distance_cm=float(np.mean([r['final_distance_cm'] for r in rows])),
        any_success_rate=float(np.mean([r['any_success'] for r in rows])),
        mean_approach_distance_cm=float(np.mean([r['approach_distance_cm'] for r in rows])),
        mean_block_displacement_cm=float(np.mean([r['block_displacement_cm'] for r in rows])), results=rows)


def selection_key(score):
    return (score['success_rate'], -score['mean_final_distance_cm'])


class Recorder(LoopCallback):
    def __init__(self, agent, pipeline, output, interval, extend_to=0):
        self.agent, self.pipeline, self.output, self.interval = agent, pipeline, output, interval
        self.started = time.monotonic()
        self.best = (-1., -1e9)
        self.last_evaluation_step = None
        self.validation_means = []
        self.extend_to = extend_to
        best_file = output/'best.json'
        if best_file.exists():
            self.best = selection_key(json.loads(best_file.read_text()))

    def checkpoint(self):
        agent = self.agent
        folder = self.output/'checkpoints'/f'{agent.steps:09d}'
        agent.save_weights(folder)
        score = evaluate(agent, VALIDATION_SEEDS)
        score.update(steps=agent.steps, checkpoint=str(folder.resolve()))
        write_json(folder/'validation.json', score)
        with (self.output/'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(score)+'\n')
        if selection_key(score) > self.best:
            self.best = selection_key(score)
            write_json(self.output/'best.json', score)
        self.last_evaluation_step = agent.steps
        self.validation_means.append((agent.steps, selection_key(score)))
        print(f"VALIDATION steps={agent.steps:,} success={score['success_rate']:.0%} error={score['mean_final_distance_cm']:.2f}cm", flush=True)
        recent = [key for step, key in self.validation_means if step > agent.steps-1_000_000]
        earlier = [key for step, key in self.validation_means if step <= agent.steps-1_000_000]
        if (self.pipeline.collected >= self.pipeline.max_steps
                and self.extend_to > self.pipeline.max_steps and earlier
                and (max(recent)[0] > max(earlier)[0] or
                     (max(recent)[0] == max(earlier)[0] and
                      max(recent)[1] > max(earlier)[1] + (0.1 if self.pipeline.approach_weight else 0.)))):
            self.pipeline.max_steps = self.extend_to
            write_json(self.output/'extension.json', dict(total_steps=self.extend_to,
                reason='Validation improved in the last million transitions; approved extension.'))
            print(f'EXTENDING to {self.extend_to:,} steps', flush=True)

    def on_episode_end(self, episode_id, eval_metrics):
        agent = self.agent
        row = dict(steps=agent.steps, updates=agent.updates,
            completed_episodes_this_session=self.pipeline.completed,
            reward_last_100=float(np.mean(self.pipeline.recent)) if self.pipeline.recent else None,
            elapsed_seconds=time.monotonic()-self.started, **agent.last_diagnostics)
        with (self.output/'training.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        write_json(self.output/'status.json', dict(state='training', **row, best_validation=self.best))
        if agent.updates % 10 == 0:
            print(f"TRAIN steps={agent.steps:,} reward100={row['reward_last_100']} kl={row['approx_kl']:.4f} elapsed={row['elapsed_seconds']:.0f}s", flush=True)
        if agent.updates % self.interval == 0 or self.pipeline.collected >= self.pipeline.max_steps:
            self.checkpoint()


def configure(device='cpu', seed=42):
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if device == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError('GPU requested but TensorFlow cannot access CUDA')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/fetch-push'))
    parser.add_argument('--steps', type=int, default=5_001_216)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--extend-to', type=int, default=10_000_384)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--approach-weight', type=float, default=0.5)
    parser.add_argument('--final-seed-base', type=int, default=400000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.steps % 2048 or 2048 % args.envs:
        parser.error('steps must be a multiple of 2048; envs must divide 2048')
    configure(args.device, args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'config.json').exists() and not args.resume:
        parser.error('Output already has an experiment; choose a new directory or --resume')
    agent = RobotAgent(args.seed)
    if args.resume:
        agent.load_weights(args.resume)
    final_seeds = list(range(args.final_seed_base, args.final_seed_base+100))
    config = dict(environment=ENV, seed=args.seed, approach_weight=args.approach_weight, display_seeds=DISPLAY_SEEDS, validation_seeds=VALIDATION_SEEDS,
        final_seeds=final_seeds, total_steps=args.steps, n_envs=args.envs, rollout_steps=2048,
        horizon_per_environment=2048//args.envs, hyperparameters=HP, device=args.device,
        tensorflow=tf.__version__, gymnasium=gym.__version__, tensor_optix=importlib.metadata.version('tensor-optix'),
        mujoco=importlib.metadata.version('mujoco'), gymnasium_robotics=importlib.metadata.version('gymnasium-robotics'),
        python=platform.python_version(), actor_parameters=agent._actor.count_params(),
        critic_parameters=agent._critic.count_params(), resume=str(args.resume) if args.resume else None,
        smoke=args.smoke)
    write_json(output/('resume-config.json' if args.resume else 'config.json'), config)
    pipeline = RobotPipeline(agent, output, args.steps, args.envs, args.seed, args.approach_weight)
    recorder = Recorder(agent, pipeline, output, args.checkpoint_every, 0 if args.smoke else args.extend_to)
    if not args.resume:
        recorder.checkpoint()
        write_json(output/'random-baseline.json', evaluate(agent, VALIDATION_SEEDS, random=True))
    opt = Optimizer(agent=agent, pipeline=pipeline, evaluator=TrainingScore(pipeline),
        optimizer=FixedParameters(), callbacks=[recorder], checkpoint_dir=str(output/'optix-internal'),
        max_snapshots=2, base_interval=1000000, max_interval_episodes=1000000,
        min_episodes_before_dormant=1000000, dormant_threshold=1000000,
        min_episodes_before_degradation=1000000, convergence_patience=1000000,
        diag_min_episodes=1000000, diag_entropy_floor=None, diag_target_kl=None)
    opt.run()
    # The Optix internal training-score snapshot is independent of our held-out
    # selection. Explicitly restore our validation-selected full state.
    selected = json.loads((output/'best.json').read_text())
    agent.load_weights(selected['checkpoint'])
    agent._actor.save(output/'model.keras')
    agent._critic.save(output/'critic.keras')
    write_json(output/'normalization.json', agent.obs_stats.state())
    if args.smoke:
        result = dict(smoke=True, evaluation=evaluate(agent, VALIDATION_SEEDS[:2]))
    else:
        result = evaluate(agent, final_seeds)
        result.update(target_achieved=result['success_rate'] >= .9, selected_steps=agent.steps,
                      trained_steps=pipeline.collected, selection='validation final-step success, then target distance; final seeds used once')
    write_json(output/'evaluation.json', result)
    write_json(output/'status.json', dict(state='complete', **result))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
