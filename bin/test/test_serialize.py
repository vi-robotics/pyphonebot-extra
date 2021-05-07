#!/usr/bin/env python3

import sys
import zlib

from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph


def main():
    graph = PhonebotGraph()
    data = graph.encode()
    print('raw')
    print(sys.getsizeof(data))
    print('compressed')
    print(sys.getsizeof(zlib.compress(data)))
    graph.restore(data)


if __name__ == '__main__':
    main()
