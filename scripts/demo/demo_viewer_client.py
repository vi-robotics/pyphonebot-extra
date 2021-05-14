#!/usr/bin/env python3

from pathlib import Path
import time
import zlib
import numpy as np

from phonebot.core.common.math.utils import anorm, alerp
from phonebot.core.common.serial import encode, decode
from phonebot.core.common.config import PhonebotSettings, FrameName

from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph.graph_utils import get_graph_geometries
from phonebot.core.common.comm.client import SimpleClient


def main():
    client = SimpleClient()
    pattern = 'state-' + '[0-9]' * 4 + '.txt'
    files = list(Path('/tmp/phonebot-phone').glob(pattern))
    for filename in sorted(files):
        with open(filename, 'rb') as f:
            print('send')
            client.send(f.read())
            time.sleep(0.1)


if __name__ == '__main__':
    main()
