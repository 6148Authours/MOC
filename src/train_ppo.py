from pathlib import Path

from causal_world.intervention_actors import GoalInterventionActorPolicy
from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.wrappers import CurriculumWrapper
from stable_baselines3 import PPO
from stable_baselines3.common import monitor
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.policies import MlpPolicy
# import tensorflow as tf
from stable_baselines3.common.monitor import Monitor

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import json
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse


def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, ppo_config, total_time_steps,
                 validate_every_timesteps, task_name):
    def _make_env(rank):
        def _init():
            task = generate_task(task_generator_id=task_name)
            env = CausalWorld(task=task,
                              skip_frame=skip_frame,
                              enable_visualization=False,
                              seed=seed_num + rank,
                              max_episode_length=maximum_episode_length)
            monitor_env = Monitor(env)
            return monitor_env

        set_random_seed(seed_num)
        return _init

    # os.makedirs(log_relative_path.as_posix(), exist_ok=True)
    # policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
    # env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    # task = generate_task(task_generator_id=task_name)
    # env = CausalWorld(task, skip_frame=skip_frame, enable_visualization=False,
    #                   max_episode_length=maximum_episode_length)
    # env = Monitor(env, "./logs/baselines")

    env = SubprocVecEnv(
        [_make_env(i) for i in range(num_of_envs)])
    model = PPO('MlpPolicy',
                env,
                _init_setup_model=True,
                verbose=1,
                **ppo_config)

    task = generate_task(task_generator_id=task_name)
    old_env = CausalWorld(task=task,
                        skip_frame=skip_frame,
                        enable_visualization=False,
                        seed=seed_num + 0,
                        max_episode_length=maximum_episode_length)
    save_config_file(ppo_config,
                     old_env,
                     os.path.join(log_relative_path.as_posix()+"_0", 'config.json'))
    for i in range(int(total_time_steps / validate_every_timesteps)):
        model.learn(total_timesteps=validate_every_timesteps,
                    tb_log_name=log_relative_path,
                    reset_num_timesteps=False)
        model.save(os.path.join(log_relative_path.as_posix()+"_0", 'saved_model'))
    return


def save_config_file(ppo_config, env, file_path):
    task_config = env._task.get_task_params()
    for task_param in task_config:
        if not isinstance(task_config[task_param], str):
            task_config[task_param] = str(task_config[task_param])
    env_config = env.get_world_params()
    env.close()
    configs_to_save = [task_config, env_config, ppo_config]
    with open(file_path, 'w') as fout:
        json.dump(configs_to_save, fout)


def evaluate_trained_policy(log_relative_path):
    model = PPO.load(os.path.join(log_relative_path, 'saved_model'))
    task = generate_task(task_generator_id=task_name)
    env = CausalWorld(task, skip_frame=skip_frame, enable_visualization=False,
                      max_episode_length=maximum_episode_length)
    # env = CurriculumWrapper(env,
    #                         intervention_actors=[GoalInterventionActorPolicy()],
    #                         actives=[(0, 1000000000, 1, 0)])
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(mean_reward, std_reward)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # TODO: pass reward weights here!!
    ap.add_argument("--seed_num", required=False, default=0, help="seed number")
    ap.add_argument("--skip_frame",
                    required=False,
                    default=10,
                    help="skip frame")
    ap.add_argument("--max_episode_length",
                    required=False,
                    default=2500,
                    help="maximum episode length")
    ap.add_argument("--total_time_steps_per_update",
                    required=False,
                    default=100000,
                    help="total time steps per update")
    ap.add_argument("--num_of_envs",
                    required=False,
                    default=2,
                    help="number of parallel environments")
    ap.add_argument("--task_name",
                    required=False,
                    default="reaching",
                    help="the task nam for training")
    ap.add_argument("--fixed_position",
                    required=False,
                    default=True,
                    help="define the reset intervention wrapper")
    ap.add_argument("--evaluate-dir", default=4, required=False, help="Evaluation dir")
    ap.add_argument('--train', default=1, type=int, help="Training the agent")
    ap.add_argument('--machine-id', default='baseline', type=str,
                    required=False, help='Machine ID')
    ap.add_argument('--device', default='cuda', type=str, help="cpu or cuda?")
    args = vars(ap.parse_args())
    total_time_steps_per_update = int(args['total_time_steps_per_update'])
    num_of_envs = int(args['num_of_envs'])
    maximum_episode_length = int(args['max_episode_length'])
    skip_frame = int(args['skip_frame'])
    seed_num = int(args['seed_num'])
    task_name = str(args['task_name'])
    fixed_position = bool(args['fixed_position'])
    train = bool(args['train'])
    device = str(args['device'])
    assert (((float(total_time_steps_per_update) / num_of_envs) /
             5).is_integer())
    if train:
        for _ in range(3):
            model_dir = Path('logs/baselines')
            if not model_dir.exists():
                run_num = 1
            else:
                exist_run_nums = [int(str(folder.name).split('run')[1].split('_')[0]) for folder in model_dir.iterdir() if
                                str(folder.name).startswith('run')]
                if len(exist_run_nums) == 0:
                    run_num = 1
                else:
                    run_num = max(exist_run_nums) + 1
            current_run = 'run{}_{}_{}'.format(run_num, str(
            args['machine_id']), str(args['task_name']))
            run_dir = model_dir / current_run
            log_dir = run_dir
            os.makedirs(log_dir.as_posix()+"_0", exist_ok=True)
            log_relative_path = log_dir
            ppo_config = {
                "gamma": 0.9995,
                "n_steps": 5000,
                "ent_coef": 0,
                "learning_rate": 0.00025,
                "vf_coef": 0.5,
                "max_grad_norm": 10,
                "device": device,
                "tensorboard_log": "./"
            }
            train_policy(num_of_envs=num_of_envs,
                        log_relative_path=log_relative_path,
                        maximum_episode_length=maximum_episode_length,
                        skip_frame=skip_frame,
                        seed_num=seed_num,
                        ppo_config=ppo_config,
                        total_time_steps=10000000,
                        validate_every_timesteps=10000000,
                        task_name=task_name)
    else:
        evaluate_dir = str(args["evaluate_dir"])
        evaluate_trained_policy("baselines/run{}".format(evaluate_dir))
