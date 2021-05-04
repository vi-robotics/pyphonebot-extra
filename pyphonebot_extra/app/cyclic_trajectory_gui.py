#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import numpy as np
import sys
import logging
from typing import Tuple

from phonebot.core.common.settings import Settings
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.controls.trajectory.trajectory_utils import parametrize_cyclic_trajectory, time_from_freq
from phonebot.core.kinematics.workspace import get_workspace

from phonebot.app.app_utils import update_settings_from_arguments


logger = logging.getLogger(__name__)


def make_cyclic(data: np.ndarray):
    """
    Add an element to wrap around to beginning of array.
    """
    return np.append(data, data[:1], axis=0)


class AppSettings(Settings):
    log_level: str
    num: int
    res: int
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    draw_workspace: bool
    robot: PhonebotSettings

    def __init__(self, **kwargs):
        self.log_level = 'INFO'
        self.num = 128
        self.res = 3
        self.xlim = (-0.05, 0.05)
        # NOTE(yycho0108): ylim here is flipped to provide
        # A more intuitive visualization for the workspace when rendered.
        self.ylim = (-0.025, 0.075)
        self.draw_workspace = True
        self.robot = PhonebotSettings()
        super().__init__(**kwargs)


class CyclicTrajectoryGui(object):
    """
    Get user input points and compute filtered cyclic trajectory.
    """

    def __init__(self, ax, settings: AppSettings):
        # Save input parameters.
        self.ax_ = ax
        self.settings_ = settings

        # Setup empty point list.
        self.xs_ = []
        self.ys_ = []

        # Setup plot handles.
        self.line_in, = ax.plot([], [], linestyle="-",
                                marker="o", color="r", label='in', picker=5)
        self.line_out, = ax.plot(
            [], [], linestyle="--", marker=".", color="b", label='out')

        # Optionally plot workspace.
        self.line_ws = None
        if settings.draw_workspace:
            ws = get_workspace(config=settings.robot)
            self.line_ws, = ax.plot(
                ws[..., 0], ws[..., 1], linestyle=':', label='workspace')

        # Attach events.
        self.cid_btn = ax.figure.canvas.mpl_connect(
            'button_release_event', self.on_click)
        self.cid_key = ax.figure.canvas.mpl_connect(
            'key_press_event', self.on_key)
        self.cid_pick = ax.figure.canvas.mpl_connect(
            'pick_event', self.on_pick)
        self.cid_move = ax.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

        self.pick_index = -1

    def on_key(self, event):
        """ Keyboard event handler """
        if event.key == ' ':
            self.optimize()

    def on_pick(self, event):
        import matplotlib
        if event.artist != self.line_in:
            return
        # On RMB : move point.
        if event.mouseevent.button == matplotlib.backend_bases.MouseButton.RIGHT:
            # right == move
            index = np.unique(event.ind % len(self.xs_))[0]
            self.pick_index = index
            return

    def on_move(self, event):
        # NOTE(ycho): Currently only intended to handle
        # vertex edits following a pick event.
        if self.pick_index < 0:
            return

        # Update vertex.
        i = self.pick_index
        self.xs_[i] = event.xdata
        self.ys_[i] = event.ydata

        # Update line.
        self.line_in.set_data(make_cyclic(self.xs_), make_cyclic(self.ys_))
        self.line_in.figure.canvas.draw()

    def on_click(self, event):
        """ Click event handler """
        import matplotlib
        if event.inaxes != self.ax_:
            return

        # On LMB : add point.
        if event.button == matplotlib.backend_bases.MouseButton.LEFT:
            self.xs_.append(event.xdata)
            self.ys_.append(event.ydata)
            self.line_in.set_data(make_cyclic(self.xs_), make_cyclic(self.ys_))
            self.line_in.figure.canvas.draw()

        # Always disable pick on release.
        print('release')
        self.pick_index = -1

    def optimize(self):
        p_in = np.stack([self.xs_, self.ys_], axis=-1)
        info = parametrize_cyclic_trajectory(
            make_cyclic(p_in), self.settings_.res, self.settings_.num)

        # Unpack parametrization output...
        Xc, fc = info
        xc, yc = Xc[..., 0], Xc[..., 1]

        # Convert to (densified) coordinates.
        t = np.linspace(0, 2*np.pi, num=128, endpoint=True)
        xx = time_from_freq(Xc[..., 0], fc, t)
        yy = time_from_freq(Xc[..., 1], fc, t)

        # Save params to property within namespace.
        self.xx, self.yy = xx, yy
        self.xc, self.yc, self.fc = xc, yc, fc

        logger.info('x:{}'.format(self.xc))
        logger.info('y:{}'.format(self.yc))
        logger.info('f:{}'.format(self.fc))

        self.line_out.set_data(self.xx, self.yy)
        self.line_out.figure.canvas.draw()

    def get_data(self):
        input_trajectory = (self.xs_, self.ys_)
        output_trajectory = (self.xx, self.yy)
        output_parameters = (self.xc, self.yc, self.fc)
        return input_trajectory, output_trajectory, output_parameters


def run_gui(settings: AppSettings):
    # NOTE(yycho0108): Importing matplotlib here
    # Just in case an unexpected dependency
    # creeps into the automated requirements setup.
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Click to add points, Space to optimize')
    ax.plot(0, 0, 'k+')
    ax.set_xlim(settings.xlim)
    ax.set_ylim(settings.ylim)
    ax.invert_yaxis()
    ax.grid()
    gui = CyclicTrajectoryGui(ax, settings)
    plt.legend()
    plt.show()
    return gui.get_data()


def main():
    # Get settings ...
    settings = AppSettings()
    update_settings_from_arguments(settings)

    # Set log level ...
    logging.basicConfig()
    logger.setLevel(settings.log_level.upper())

    data = run_gui(settings)
    logger.debug(data)


if __name__ == '__main__':
    main()
