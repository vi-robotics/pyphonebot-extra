#!/usr/bin/env python3
"""
WARN(yycho0108):
Simulation with Mujoco backend is no longer supported.
It may still be functional after a couple of patches, but there is no means for me to test it.
Leaving the code here for legacy purposes.
"""

import numpy as np
import os
from pathlib import Path

from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register
import numpy as np

from phonebot.core.common.config import PhonebotSettings
from phonebot.core.kinematics import PhonebotKinematics
from phonebot.core.common.path import PhonebotPath


def _get_joint_names_map(config: PhonebotSettings):
    jtmap = dict(knee='j', hip='m')
    jmap = dict()
    for prefix in config.order:
        for suffix in 'ab':
            for jtype in ['knee', 'hip']:
                # model def
                j_from = '{}_{}_joint_{}'.format(prefix, jtype, suffix)
                # asset def
                j_to = '{}_m{}'.format(prefix.lower(), jtmap[jtype], suffix)
                jmap[j_from] = j_to
    return jmap


class PhoneBotEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.pk_ = PhonebotKinematics()
        cfg_xml = Path(PhonebotPath.assets()) / 'phonebotx1.xml'
        mujoco_env.MujocoEnv.__init__(self, cfg_xml, 25)
        utils.EzPickle.__init__(self)
        print(dir(self.model))

        # NOTE: joint names
        #['root', 'fl_ma', 'fl_ja', 'fl_mb', 'fl_jb', 'fr_ma', 'fr_ja', 'fr_mb', 'fr_jb', 'bl_ma', 'bl_ja', 'bl_mb', 'bl_jb', 'br_ma', 'br_ja', 'br_mb', 'br_jb']

        # 7 elem root pose : [xyz] + [qw,qx,qy,qz]
        qroot = [0, 0, 0.2, 1, 0, 0, 0]
        #qroot = []
        # 8x2 joint positions, roughly set to valid-ish position
        qjoints = np.repeat([[1.40429062, -2.4490134]], 8, axis=0).ravel()
        p0 = np.r_[qroot, qjoints].ravel()
        self.rectify_qpos(p0)
        self.init_qpos = p0
        self.init_qvel = np.zeros(22)

    def _step(self, a):
        # TODO : return better cost
        xposbefore = self.get_body_com("phone")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("phone")[0]
        forward_reward = 1.0 * (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        #reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        reward = forward_reward
        state = self.state_vector()
        #notdone = np.isfinite(state).all() \
        #    and state[2] >= 0.2 and state[2] <= 1.0
        #done = not notdone
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        # TODO : return better observation parameters
        # return np.concatenate([
        #    self.model.data.qpos.flat[2:],
        #    self.model.data.qvel.flat,
        #    np.clip(self.model.data.cfrc_ext, -1, 1).flat,
        # ])

        # TODO : maybe revert at some point
        xpos = self.get_body_com("phone")[0]
        js = self.model.data.qpos[7:, 0]
        t = self.data.time
        return np.concatenate(
            [[t, xpos], js])

    def rectify_qpos(self, qpos):
        # NOTE : in-place modification

        # TODO : make this more robust (works, but lots of magic indices)
        for i in range(7, 7+2*2*4, 2*2):
            # may be reversed
            qpos[i+1], qpos[i+3] = self.pk_.solve_passive(
                h1=qpos[i], h2=qpos[i+2],
                seed={'h3': qpos[i+1], 'h4': qpos[i+3]})

    def reset_model(self):
        # print(self.model.nq)
        qpos = self.init_qpos + \
            self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        self.rectify_qpos(qpos)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


# TODO : provide proper num episodes + reward thresh
register(
    id='PhoneBot-v0',
    entry_point='sim.phonebot_env:PhoneBotEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0
)
