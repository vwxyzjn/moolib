# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gym
import envpool
import numpy as np

from . import atari_preprocessing


# def create_env(flags):
#     env = gym.make(  # Cf. https://brosa.ca/blog/ale-release-v0.7
#         flags.env.name,
#         obs_type="grayscale",  # "ram", "rgb", or "grayscale".
#         frameskip=1,  # Action repeats. Done in wrapper b/c of noops.
#         repeat_action_probability=flags.env.repeat_action_probability,  # Sticky actions.
#         full_action_space=True,  # Use all actions.
#         render_mode=None,  # None, "human", or "rgb_array".
#     )

#     # Using wrapper from seed_rl in order to do random no-ops _before_ frameskipping.
#     # gym.wrappers.AtariPreprocessing doesn't play well with the -v5 versions of the game.
#     env = atari_preprocessing.AtariPreprocessing(
#         env,
#         frame_skip=flags.env.num_action_repeats,
#         terminal_on_life_loss=False,
#         screen_size=84,
#         max_random_noops=flags.env.noop_max,  # Max no-ops to apply at the beginning.
#     )
#     env = gym.wrappers.FrameStack(env, num_stack=4)
#     return env


class EnvPoolSingleEnvWrapepr:
    def __init__(self, env) -> None:
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, reward, done, info = self.env.step(np.array([action]))
        return obs[0], reward[0], done[0], {}

ATARI_MAX_FRAMES = int(
    108000 / 4
)  # 108000 is the max number of frames in an Atari game, divided by 4 to account for frame skipping


def create_env(flags):
    envs = envpool.make(
        flags.env.name,
        env_type="gym",
        num_envs=1,
        episodic_life=False,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 6
        repeat_action_probability=0.25,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12
        noop_max=1,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12 (no-op is deprecated in favor of sticky action, right?)
        full_action_space=True,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) Tab. 5
        max_episode_steps=ATARI_MAX_FRAMES,  # Hessel et al. 2018 (Rainbow DQN), Table 3, Max frames per episode
    )
    envs = EnvPoolSingleEnvWrapepr(envs)
    return envs

