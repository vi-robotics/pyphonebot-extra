#!/usr/bin/env python3

"""
Plot gait profile.
Typically to be used on data that has been produced
by PybulletPhonebotEnv with settings.debug_contact == `True`.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from simple_parsing import ArgumentParser


from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


@dataclass
class Settings:
    data_file: str = '/tmp/contact.txt'  # contact data csv.


def consecutive(data, stepsize=1):
    """
    Determine a contiguous interval in a series,
    where contiguity is defined by matching against `stepsize`.

    https://stackoverflow.com/a/7353335
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def main():
    # Parse arguments.
    parser = ArgumentParser()
    parser.add_arguments(Settings, dest='opts')
    opts = parser.parse_args().opts

    # Read df.
    df = pd.read_csv(opts.data_file, delimiter=' ',
                     names=['time_index', 'link_name'])

    # Map link names to link indices.
    link_index_from_name = {'FL': 1, 'FR': 2, 'HL': 3, 'HR': 4}
    link_id = np.zeros_like(df.link_name, dtype=np.int32)
    for link_name, link_index in link_index_from_name.items():
        link_id[[link_name in l for l in df.link_name]] = link_index

    # Add link index to dataframe.
    df['link_id'] = link_id

    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    count = 0
    for link_id, df in df.groupby('link_id'):
        # NOTE(ycho): Skip unassigned contacts (e.g. `body`)
        if link_id == 0:
            continue

        # Get contact times and convert them to intervals.
        contact_times = np.unique(np.sort(df.time_index.to_numpy()))
        contact_intervals = consecutive(contact_times)

        # Convert contact intervals into line segments.
        segs = np.zeros((len(contact_intervals), 2, 2), dtype=np.float32)
        for i, interval in enumerate(contact_intervals):
            rhs = np.asarray([[interval[0], link_id],
                              [interval[-1], link_id]], dtype=np.float32)
            segs[i, :, :] = rhs

        # Add line segments to the collection.
        plt.gca().add_collection(
            LineCollection(segs, linewidth=20, colors=colors[count]))
        count += 1

    # Set axes limits.
    plt.gca().set_ylim(1 - 2, 4 + 2)
    plt.gca().set_xlim(df.time_index.min(), df.time_index.max())

    # Set axes styles.
    plt.title('Phonebot Gait Profile')
    plt.gca().set_yticks(list(link_index_from_name.values()))
    plt.gca().set_yticklabels(list(link_index_from_name.keys()))
    plt.xlabel('Time index')
    plt.grid()

    # Finally, display the gait profile.
    plt.show()


if __name__ == '__main__':
    main()
