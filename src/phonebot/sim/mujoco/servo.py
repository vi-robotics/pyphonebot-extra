""" Positional Servo Controller """
import sys
import gym
import time
import numpy as np
from multiprocessing import Pool
from functools import partial

from phonebot.core.controls import PID
from phonebot.core.kinematics import PhonebotKinematics
# from sim.phonebot_env import PhoneBotEnv


def anorm(x):
    return (x + np.pi) % (2*np.pi) - np.pi


class Servo(object):
    def __init__(self, env,
                 kp=0.7911498954868115,
                 ki=0.038686981533994094,
                 kd=0.006905904664440783
                 ):
        self.env_ = env
        self.pid_ = [PID(kp, ki, kd) for _ in range(8)]

    def reset(self):
        [p.reset() for p in self.pid_]

    def __call__(self, target, dt, return_err=False, qp=None):
        if qp is None:
            if self.env_ is None:
                raise ValueError("qp must be set if self.env_==None")
            qp = self.env_.unwrapped.model.data.qpos[7:-1:2].ravel()
        err = anorm(target - qp)
        ctrl = [p(e, dt) for (p, e) in zip(self.pid_, err)]
        if return_err:
            return ctrl, err
        return ctrl


def tune_servo(params, env, max_t=200):
    servo = Servo(env, *params)
    servo.reset()
    env.reset()
    t0 = env.env.data.time
    net_err = 0.0

    qp = env.unwrapped.model.data.qpos[7:-1:2].ravel()
    ctrl = np.random.uniform(0, np.pi/2, size=8)
    init_err = np.square(qp - ctrl).mean()

    for i in range(max_t):
        t = env.env.data.time
        dt = t - t0
        t0 = t
        # ctrl = np.full(8, 0.5)#np.zeros(8)
        cmd, err = servo(ctrl, dt, return_err=True)
        net_err += np.square(err).mean()  # - init_err
        done = env.step(cmd)[2]
    return float(net_err) / max_t


def tune_servo_multi(envs, params, *args, **kwargs):
    p = Pool(4)
    res = np.mean(p.map(partial(tune_servo, params), envs, chunksize=1))
    p.close()
    p.join()
    print(res)
    return res


def main():
    # == tune twiddle ==
    #from twiddle import Twiddle
    #params = [1.1980000000000004*0.25, 0.02635380000000002, 0.001650194885526641*4]
    #envs = [gym.make('PhoneBot-v0') for _ in range(4)]
    #twiddle_joint = Twiddle(partial(tune_servo_multi,envs), params)
    # for _ in range(100):
    #    (iterations, params, best_run) = twiddle_joint.run()
    #    print 'n_it', iterations
    #    print 'params', params
    #    print 'best', best_run

    # == test servo ==
    # -- initialize --
    env = gym.make('PhoneBot-v0')
    servo = Servo(env)
    env.reset()
    servo.reset()
    t0 = env.env.data.time
    done = False

    ctrl = np.full(8, 0.5)
    errs = []
    for i in range(400):
        if done:
            env.reset()
            servo.reset()
            t0 = env.env.data.time
        env.render()
        t = env.env.data.time
        dt = t - t0
        t0 = t
        # print 'ctrl', ctrl
        qp = env.unwrapped.model.data.qpos[7:-1:2].ravel()
        err = anorm(ctrl - qp)  # normalized angular error
        #err = np.abs(err).mean()
        errs.append(err)
        #print(i, 'err', err)
        # ctrl = np.full(8, 0.2)#np.zeros(8)
        cmd = servo(ctrl, dt)
        #print('cmd', cmd)
        done = env.step(cmd)[2]
        time.sleep(0.001)
    np.save('/tmp/err.npy', errs)


if __name__ == "__main__":
    main()
