#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argcomplete
import argparse
import json
from typing import get_type_hints
from typing import List, Dict

from phonebot.core.common.settings import Settings


def build_parser(settings: Settings,
                 root: argparse.ArgumentParser,
                 name: str = ''):
    """Recursive function which adds an argument parsers to a provided root
    parser which parses all settings configured in a Settings object. If another
    Settings instance is a value, then a parser group is made with that Settings
    object and added to the parser as well.

    Args:
        settings (Settings): The Settings object to get the argument parser for
        root (argparse.ArgumentParser): The argument parser to add additional
            parsers onto.
        name (str, optional): The name of the current parser group. Defaults to
            ''.
    """
    group = root.add_argument_group(name, settings.__doc__)
    for key, value in settings.__dict__.items():
        if isinstance(value, Settings):
            if len(name) <= 0:
                build_parser(value, root, key)
            else:
                build_parser(value, root, '{}.{}'.format(name, key))
        else:
            type_hint = settings.__annotations__[key]
            # TODO(yycho0108): Remove this workaround.
            parse_fn = str if (type_hint == str) else json.loads
            if len(name) <= 0:
                group.add_argument(F'--{key}',
                                   default=value, type=parse_fn,
                                   help=F'{key}({type_hint})')
            else:
                group.add_argument(F'--{name}.{key}',
                                   default=value, type=parse_fn,
                                   help=F'{key}({type_hint})')


def update_settings_from_arguments(settings: Settings) -> Settings:
    """Update settings from arguments (in-place)

    Args:
        settings (Settings): Update the values of the Settings argument by
            parsing the arguments passed from command line.

    Returns:
        Settings: The updated Settings object.
    """

    # Setup settings and arguments.
    root = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    build_parser(settings, root)
    argcomplete.autocomplete(root)

    # Parse arguments and update in-place.
    args = root.parse_args()
    args = vars(args)
    settings.update(args)
    return settings


class SampleData():
    def __init__(self):
        self.x = 15


def main():

    class SampleSettings(Settings):
        """
        Sample Settings.
        """
        values: List[str]
        hmm: Dict[str, int]

        def __init__(self, **kwargs):
            self.values = ['a', 'b']
            self.hmm = {'b': 4}
            #self.xxx = SampleSettings.xxx
            super().__init__(**kwargs)

        @property
        def name(self):
            return "sample"

        @property
        def description(self):
            return "Example settings"

    class RootSettings(Settings):
        """
        Root Settings.
        """
        global_value: SampleData

        def __init__(self, **kwargs):
            self.global_value = SampleData()
            self.sample: SampleSettings = SampleSettings(values=[1, 2])
            self.sample2: SampleSettings = SampleSettings(values=[1, 2])
            super().__init__(**kwargs)

        @property
        def name(self):
            return ''

    # 1 - Instantiate (optionally, from file)
    settings = RootSettings()
    # settings = RootSettings.from_file('/tmp/settings.json')
    # print(RootSettings.from_string(f'{settings}'))

    # 2 - Load from file
    # settings.load('/tmp/settings.json')

    # 3 - Prepare argparse
    root = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    build_parser(settings, root)
    argcomplete.autocomplete(root)

    # 4 - Commit argparse results
    args = root.parse_args()
    print(F'args:{args}')
    settings.update(vars(args))

    # 5 - Save
    settings.save('/tmp/settings.json')


if __name__ == '__main__':
    main()
