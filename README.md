# pyphonebot_extra

This repository contains all of the non-core software for pyphonebot, including
but not limited to simulators, visualizers and RL agents.

## Getting started

Refer to the [wiki page](https://github.com/yycho0108/PhoneBot/wiki/Getting-Started) for instructions regarding setup.

## Sample Applications

All of sample applications are in `phonebot/app/`. The intent is to keep the apps runnable and up-to-date.

Note that unit-tests are separately maintained in `phonebot/tests`.

- Cyclic Trajectory Gui

  [cyclic_trajectory_gui.py](phonebot/app/cyclic_trajectory_gui.py)

  Sample application for generatic a parametric trajectory in the frequency domain.

  `python3 -m phonebot.app.cyclic_trajectory_gui`

  Note that here, and afterwards, the script may be directly invoked if the package is installed globally.

- Graph Viewer

  [demo_graph_viewer.py](phonebot/app/demo_graph_viewer.py)

  Sample application for using the graph viewer. Currently sweeps through all joint angles

  For testing forward and inverse kinematics as implemented on phonebot.

  `python3 -m phonebot.app.demo_graph_viewer`

- Genetic Algorithm

  [demo_genetic_algorithm.py](phonebot/app/demo_genetic_algorithm.py)

  Sample application for running a genetic algorithm for trajectory optimization.

  Prior to the port, the GA generated a walking trajectory.

  After the naive port, it is too slow and have not been verified to work.

  `python3 -m phonebot.app.demo_genetic_algorithm #--help`

- Pybullet

  [demo_pybullet.py](phonebot/app/demo_pybullet.py)

  Sample application for the pybullet simulator. Currently runs an elliptical walking trajectory.

  `python3 -m phonebot.app.demo_pybullet`

- Rotation Controller

  [demo_rotation_controller.py](phonebot/app/demo_rotation_controller.py)

  Sample application for running a rotation controller for the base orientation (i.e. roll and pitch).

  `python3 -m phonebot.app.demo_rotation_controller`

- Walker

  [demo_walker.py](phonebot/app/demo_walker.py)

  Sample application for open-loop walking trajectory demo, without simulation bindings.

  `python3 -m phonebot.app.demo_walker`

- Agent Runner

  [run_agent.py](phonebot/app/run_agent.py)

  Sample application for running an arbitrary agent.

  `python3 -m phonebot.app.run_agent --agent_type ellipse_agent`

## Appendix

- [Package Structure](docs/structure.md)
  Display of the complete package structure.
