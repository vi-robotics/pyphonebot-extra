#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

"""
Genetic Algorithm port from an old implementation.
The app is able to be run, but testing configuration has not been completed.
Also, no optimization was performed (runs very very slowly).
Most critically, mutation does not work due to the failure of controllers
to handle out-of-bounds trajectories.
Mostly here as a reference.
"""

import sys
import os
import time
import gym
import pickle
import numpy as np
import multiprocessing as mp
import logging

from enum import Enum
from typing import Tuple
from functools import partial

from phonebot.core.common.math.transform import Position
from phonebot.core.common.settings import Settings
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.controls.trajectory import WaypointTrajectory
from phonebot.core.controls.trajectory.trajectory_utils import time_from_freq, get_elliptical_trajectory, parametrize_cyclic_trajectory
from phonebot.core.kinematics.workspace import get_workspace, max_rect
from phonebot.core.controls.agents.trajectory_agent import TrajectoryAgentGraph
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph

from phonebot.sim.pybullet.simulator import PybulletPhonebotEnv, PybulletSimulatorSettings
from phonebot.app.cyclic_trajectory_gui import run_gui, AppSettings as CyclicTrajectoryGuiSettings

from phonebot.app.app_utils import update_settings_from_arguments


class InitPolicy(Enum):
    """
    Options for population initialization.
    """
    MANUAL = 0
    RANDOM = 1
    ELLIPSE = 2


class GeneticAlgorithmSettings(Settings):
    init_policy: InitPolicy
    trajectory_gui: CyclicTrajectoryGuiSettings()

    initial_std: Tuple[float, float, float]
    mutation_std: Tuple[float, float, float]
    cross_rate: float
    num_iter: int
    max_time: int
    pop_size: int
    debug: bool
    period: float

    def __init__(self, **kwargs):
        self.init_policy = InitPolicy.ELLIPSE
        self.trajectory_gui = CyclicTrajectoryGuiSettings()
        self.initial_std = (0.0002, 0.0002, 2.0 * np.pi)
        self.mutation_std = (0.00001, 0.00001, 0.03)
        self.cross_rate = 0.8
        self.num_iter = 128
        self.max_time = 512
        self.pop_size = 128
        self.debug = True
        self.period = 2.0
        super().__init__(**kwargs)


class AppSettings(Settings):
    test: bool
    train: bool
    genetic_algorithm: GeneticAlgorithmSettings
    sim: PybulletSimulatorSettings
    robot: PhonebotSettings
    num_envs: int

    def __init__(self, **kwargs):
        self.test = True
        self.train = True
        self.genetic_algorithm = GeneticAlgorithmSettings()
        self.sim = PybulletSimulatorSettings()
        self.sim.render = False
        self.num_envs = 1
        self.robot = PhonebotSettings()


def crossover(population: np.ndarray, cross_rate=0.8, copy=False):
    if copy:
        population = np.copy(population)

    # mating process (genes crossover)
    n, m = population.shape
    k = max(1, int(n * cross_rate))

    indices = np.random.choice(n, size=(k, 2))
    parents = population[indices]
    population[indices[:, 0]] = np.mean(parents, axis=1)

    return population


def mutate(
        population: np.ndarray, freq: np.ndarray,
        std: Tuple[float, float, float],
        num_out=None):
    # population = [Nx4xMxD], M = nfreq, D=dims
    population = np.asarray(population)
    num_in = population.shape[0]
    if num_out is None:
        num_out = num_in
    nfreq = freq.shape[0]

    # 0. Unroll standard deviation parameter.
    mstd, astd, ostd = std

    # 1. Determine mutation source from population.
    source_index = np.random.choice(num_in, size=num_out)
    source = population[source_index]

    # 2. Mutate magnitude.
    mag = np.random.normal(
        loc=np.abs(source),
        scale=mstd)

    # 3. Mutate Angle.
    ang = np.random.normal(
        scale=astd,
        size=source.shape)

    # 3-1. Apply frequency to mutation.
    ang = np.einsum('abcd,c->abcd', ang, freq)

    # 3-2. Apply jointwise phase-offset.
    phase = np.random.normal(
        scale=ostd * (freq / np.max(freq)),
        size=(num_out, 4, nfreq))[..., None]
    ang += phase

    # 3-3. Apply increment to source.
    ang = np.angle(source) + ang

    # 3. convert back to complex-encoded coefficient form
    rpart = mag * np.cos(ang)
    ipart = mag * np.sin(ang)

    return rpart + 1j * ipart


def select(params: np.ndarray, scores: np.ndarray, num_out=None):
    # Sanitize args ...
    params = np.asarray(params)
    scores = np.asarray(scores)
    # Determine selection arguments ...
    num_in = params.shape[0]
    if num_out is None:
        num_out = num_in

    # Apply natural selection w.r.t population's fitness.
    # TODO : Better sampling metric.
    min_score = np.min(scores)
    if min_score < 0:
        scores -= min_score
    scores += 1e-3
    scores /= np.sum(scores)

    index = np.random.choice(num_in, size=num_out, replace=True,
                             p=scores)
    return params[index]


def evaluate(
        agent: TrajectoryAgentGraph, graph: PhonebotGraph,
        env: PybulletPhonebotEnv, config: PhonebotSettings, timestep: float,
        max_t=100):
    stamp = 0.0
    done = False
    score = 0.0
    state = env.reset()
    i = 0
    while not done:
        # Sense-Think-Act
        update_frame_graph(graph, env, stamp, config)
        command = agent(state, stamp)
        state, reward, done, info = env.step(command)

        # Update iteration data.
        score += reward
        state += timestep
        i += 1
        if i >= max_t:
            break

    return score


def update_frame_graph(
        graph: PhonebotGraph, env: PybulletPhonebotEnv, stamp: float,
        config: PhonebotSettings):
    joint_states = env.sense()
    js = {key: value for key, value in zip(
        config.joint_names, joint_states)}
    for prefix in config.order:
        for suffix in 'ab':
            hip_joint = '{}_hip_joint_{}'.format(prefix, suffix)
            knee_joint = '{}_knee_joint_{}'.format(prefix, suffix)
            foot_joint = '{}_foot_{}'.format(prefix, suffix)
            graph.get_edge(knee_joint, hip_joint).update(
                stamp, js[hip_joint])
            graph.get_edge(foot_joint, knee_joint).update(
                stamp, js[knee_joint])


class WaypointTrajectoryWrapper(WaypointTrajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, stamp):
        out = super().evaluate(stamp)
        return Position([out[0], out[1], 0])


def get_trajectories(param, freq, period=2.0):
    angles = np.linspace(0, 2 * np.pi, 128)
    times = np.linspace(0, 2.0, 128)
    out = []
    for xcoef, ycoef in param.reshape(4, 2, -1):
        xvals = time_from_freq(xcoef, freq, angles)
        yvals = time_from_freq(ycoef, freq, angles)
        trajectory = np.stack([xvals, yvals], axis=-1)
        out.append(WaypointTrajectoryWrapper(times, trajectory, period))
    return out


def train_ga(settings: AppSettings):
    # Parse input settings.
    ga = settings.genetic_algorithm
    config = settings.robot

    # Initialize parameters.
    if ga.init_policy == InitPolicy.MANUAL:
        _, _, init_param = run_gui(ga.trajectory_gui)
    elif ga.init_policy == InitPolicy.RANDOM:
        res = 8
        xcoef = np.random.normal(0.0, 1.0, size=res) + \
            1j * np.random.normal(0.0, 1.0, size=res)
        ycoef = np.random.normal(0.0, 1.0, size=res) + \
            1j * np.random.normal(0.0, 1.0, size=res)
        freqs = np.fft.rfftfreq(res, 1.0 / (res - 1))
        freq = np.random.choice(freqs, size=cres)
        init_param = (xcoef, ycoef, freq)
    elif ga.init_policy == InitPolicy.ELLIPSE:
        # Workspace -> Inscribed Trajectory
        workspace = get_workspace(0.0, config, return_poly=True)
        rect = max_rect(workspace, 4096)
        (x0, y0), (x1, y1) = rect
        rect = (0.5 * (x0 + x1), 0.5 * (y0 + y1),
                abs(x1 - x0), abs(y1 - y0))
        trajectory = get_elliptical_trajectory(
            rect, 2 * np.pi, 0.0, False, 1.0)

        # Trajectory -> Points -> Parameter
        # NOTE(yycho0108): Period at 2*pi is required as an implicit assumption
        # from time_from_freq.
        times = np.linspace(0.0, 2 * np.pi, 128, endpoint=True)
        points = trajectory.evaluate(times)
        param0, freq = parametrize_cyclic_trajectory(
            points[..., :2], 5, 128, times)
    param0 = np.repeat(param0[None, None, ...], 4, axis=1)  # 1x4xMx3
    nfreq = len(freq)
    traj_from_param = partial(get_trajectories, freq=freq, period=ga.period)

    # Generate population from initialization.
    params = mutate(param0, freq, ga.initial_std,
                    num_out=ga.pop_size)
    if ga.debug:
        # Show the trajectory derived from the first mutated parameter.
        from matplotlib import pyplot as plt
        param = params[0]
        logging.debug(param.shape)
        coefs = param.reshape(4, nfreq, 2)
        plt.plot(points[..., 0], points[..., 1], '+-')
        for coef in coefs:
            logging.debug(coef.shape)
            logging.debug(freq.shape)
            angles = np.linspace(0, 2 * np.pi, endpoint=True)
            xvals = time_from_freq(coef[..., 0], freq, angles)
            yvals = time_from_freq(coef[..., 1], freq, angles)
            plt.plot(xvals, yvals, 'x-')
            plt.show()
            break

    # TODO(yycho0108): Consider reviving vectorized env.
    # envs = SubprocVecEnv(env_fns=[lambda:gym.make('PhoneBot-v2') for _ in range(pop_size) ])
    envs = [
        gym.make(
            'phonebot-pybullet-v0',
            sim_settings=settings.sim,
            phonebot_settings=config) for _ in range(
            settings.num_envs)]

    np.save('/tmp/ga-fr.npy', freq)

    best_score = -np.inf
    best_param = None

    for i in range(ga.num_iter):
        # Run simulation based on current population.
        scores = []
        for isplit in range(0, len(params), len(envs)):
            sparams = params[isplit:isplit + len(envs)]

            # Initialize all handles with new param/env ...
            # TODO(yycho0108): Implement proper reset to reuse handles.
            graphs = [PhonebotGraph(config) for _ in envs]
            agents = [
                TrajectoryAgentGraph(
                    graph, ga.period, config,
                    {k: v for k,
                     v in zip(config.order, traj_from_param(param))})
                for graph, param in zip(graphs, sparams)]
            sscores = [
                evaluate(
                    agent,
                    graph,
                    env,
                    config,
                    settings.sim.timestep,
                    ga.max_time) for (
                    agent,
                    graph,
                    env) in zip(
                    agents,
                    graphs,
                    envs)]
            scores.extend(sscores)

        # Re-populate based on scores.
        params = select(params, scores)
        params = crossover(params, ga.cross_rate)
        params = mutate(params, freq, ga.mutation_std, ga.pop_size)

        # stat + log
        sel = np.argmax(scores)
        np.save('/tmp/ga-pop-{}.npy'.format(i),
                params)  # save entire population
        np.save('/tmp/ga-cur-{}.npy'.format(i),
                params[sel])  # save current best

        # Save intermediate best param.
        if best_score < scores[sel]:
            best_score = scores[sel]
            best_param = params[sel]
            np.save('/tmp/ga-best.npy', best_param)  # update overall "best"

        print('{}/{} best score : {}'.format(i, ga.num_iter, best_score))
        print('best in current generation : {}'.format(scores[sel]))

    np.save('/tmp/ga-result.npy', params)


def test_ga():
    from matplotlib import pyplot as plt
    # param0, msk = draw_path()
    # param0 = np.repeat([param0], 4, axis=0).reshape(-1) # make it identical for 4 legs
    # params = mutate(param0, 0.01, n=10)

    # for p in params:
    #    traj = generate_trajectory_from_param(p, msk)
    #    for t in traj:
    #        plt.plot(t[:1,0], t[:1,1], 'o')
    #        plt.plot(t[:,0], t[:,1])
    # plt.axis('equal')
    # plt.show()

    # x = np.load('/tmp/ga-cur-1.npy')
    x = np.load('/tmp/ga-best.npy')

    msk = np.load('/tmp/ga-msk.npy')
    print('msk', msk)
    # msk  = np.zeros(101, dtype=np.bool)
    # msk[:3] = 1
    traj = generate_trajectory_from_param(x, msk)
    for traj_, label, col in zip(traj, ['fl', 'fr', 'bl', 'br'], 'rgby'):
        plt.plot(traj_[:1, 0], traj_[:1, 1], 'o', color=col)
        plt.plot(traj_[:, 0], traj_[:, 1], label=label, color=col)
    plt.legend()
    plt.show()
    # env = gym.make('PhoneBot-v0')
    agent = TrajectoryFollowerAgent(env, period=4.0)

    obs = env.reset()
    agent.reset(traj)

    t = obs[0]
    x0 = obs[1]
    js = obs[2:]

    for i in range(2000):
        cmd = agent.control(js, t)
        obs, _, done, _ = env.step(cmd)
        t = obs[0]
        x0 = obs[1]
        js = obs[2:]
        if done:
            break
        env.render()


def main():
    settings = AppSettings()
    update_settings_from_arguments(settings)

    if settings.train:
        train_ga(settings)

    if settings.test:
        test_ga(settings)

    test_ga()


if __name__ == "__main__":
    main()
