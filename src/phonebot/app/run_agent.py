#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import argcomplete

import gym
import time
from typing import List, Dict
from collections import defaultdict
import pybullet as pb
import logging
import numpy as np

from phonebot.core.common.settings import Settings
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.logger import set_log_level, get_default_logger
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph

from phonebot.core.controls.agents.trajectory_agent.trajectory_agent import TrajectoryAgentGraph
from phonebot.core.controls.agents.random_agent.random_agent import RandomAgent, RandomAgentSettings
from phonebot.core.controls.agents.keyboard_agent.keyboard_agent import KeyboardAgent
from phonebot.core.controls.agents.keyboard_agent.joint_keyboard_agent import JointKeyboardAgent, JointKeyboardAgentSettings
from phonebot.core.controls.agents.keyboard_agent.ellipse_keyboard_agent import EllipseKeyboardAgent, EllipseKeyboardAgentSettings

from phonebot.sim.pybullet.builder import PybulletBuilder
from phonebot.sim.pybullet.simulator import PybulletPhonebotEnv, PybulletSimulatorSettings

from phonebot.app.app_utils import update_settings_from_arguments

logger = get_default_logger()

agent_classes = {
    'key_agent': JointKeyboardAgent,
    'ellipse_agent': EllipseKeyboardAgent,
    'random_agent': RandomAgent
}


class AgentSettingsRegistry(Settings):
    """
    Agent settings registry.
    """
    key_agent: JointKeyboardAgentSettings
    ellipse_agent: EllipseKeyboardAgentSettings
    random_agent: RandomAgentSettings

    def __init__(self, **kwargs):
        self.key_agent = JointKeyboardAgentSettings()
        self.ellipse_agent = EllipseKeyboardAgentSettings()
        self.random_agent = RandomAgentSettings()
        super().__init__(**kwargs)

    def get_settings(self, agent_type):
        try:
            return getattr(self, agent_type)
        except AttributeError as e:
            logger.error('Agent Type {} unknown; valid options : {}'.format(
                agent_type, list(agent_classes.keys())))
            raise


class AppSettings(Settings):
    """
    App settings for running pybullet simulator with phonebot.
    Specifically, enables executing an arbitrary agent.
    """
    sim: PybulletSimulatorSettings
    robot: PhonebotSettings
    agent_settings: AgentSettingsRegistry
    agent_type: str
    use_viewer: bool
    log_level: str

    def __init__(self, **kwargs):
        self.sim = PybulletSimulatorSettings()
        self.robot = PhonebotSettings()
        self.agent_settings = AgentSettingsRegistry()
        self.agent_type = 'key_agent'
        self.use_viewer = True
        self.log_level = 'WARN'
        super().__init__(**kwargs)


def main():
    # Setup settings and arguments.
    settings = AppSettings()
    update_settings_from_arguments(settings)
    settings = AppSettings.from_string(str(settings))
    set_log_level(settings.log_level)
    logger.setLevel(settings.log_level)
    logging.basicConfig()

    logger.debug('{}'.format(settings))
    config = settings.robot
    agent_settings = settings.agent_settings.get_settings(settings.agent_type)

    # Instantiate all handles (env/agent/graph...)
    env = gym.make('phonebot-pybullet-v0', sim_settings=settings.sim,
                   phonebot_settings=settings.robot)
    state = env.reset()
    graph = PhonebotGraph(settings.robot)
    # FIXME(yycho0108): Remove special handling of certain agents.
    agent_cls = agent_classes[settings.agent_type]
    if agent_cls == EllipseKeyboardAgent:
        agent = agent_cls(graph, settings.robot, agent_settings)
    else:
        agent = agent_cls(settings.robot, agent_settings)

    # Disable default keyboard shortcuts
    # If keyboard events are to be sourced directly from pybullet.
    use_pybullet_key_event = isinstance(
        agent, KeyboardAgent) and (agent.event_source_ is None)
    if use_pybullet_key_event:
        pb.configureDebugVisualizer(
            flag=pb.COV_ENABLE_KEYBOARD_SHORTCUTS, enable=0,
            physicsClientId=env.sim_id
        )

    stamp = 0.0
    while True:
        stamp += settings.sim.timestep

        # Update joint states.
        # TODO(yycho0108): Reduce duplicate code usage everywhere.
        state = env.sense()
        active_joint_states = state[env.sensor.slice_active_joints]
        passive_joint_states = state[env.sensor.slice_passive_joints]
        joint_states = np.zeros(len(config.joint_names), dtype=np.float32)
        joint_states[config.active_joint_indices] = active_joint_states
        joint_states[config.passive_joint_indices] = passive_joint_states
        joint_states_map = {key: value for key, value in zip(
            settings.robot.joint_names, joint_states)}
        for prefix in settings.robot.order:
            for suffix in 'ab':
                hip_joint = '{}_hip_joint_{}'.format(prefix, suffix)
                knee_joint = '{}_knee_joint_{}'.format(prefix, suffix)
                foot_joint = '{}_foot_{}'.format(prefix, suffix)
                graph.get_edge(knee_joint, hip_joint).update(
                    stamp, joint_states_map[hip_joint])
                graph.get_edge(foot_joint, knee_joint).update(
                    stamp, joint_states_map[knee_joint])

        # Keyboard event callback.
        # Since event_source was `None`, agent.on_key()
        # Must be invoked manually.
        if use_pybullet_key_event:
            keys = pb.getKeyboardEvents(env.sim_id)
            keys = [chr(key) for (key, state) in keys.items() if (
                chr(key) in agent_settings.key_map) and (state & pb.KEY_IS_DOWN)]
            for key in keys:
                agent.on_key(key)

        # Apply control output from agent.
        ctrl = agent(state, stamp)
        state = env.step(ctrl)[0]

    env.close()


if __name__ == '__main__':
    main()
