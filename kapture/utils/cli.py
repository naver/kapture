# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Command line helpers shared by all kapture tools.

A tool is a plain, type-hinted function with a reST docstring; its command line is derived from it:

.. code-block:: python

    def import_something(input: str, output: str, force: bool = False,
                         verbose: Verbosity = 'warning', quiet: bool = False) -> None:
        \"\"\"
        One line description shown at the top of the help.

        :param input: path to the something to import
        :param output: path to the output kapture
        :param force: delete output if already exists
        :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
        :param quiet: silence all logs
        \"\"\"
        setup_logging(logger, verbose, quiet)
        ...

    def main():
        run_cli(import_something, aliases={'--output': ['-k', '--kapture']})

Conventions applied by :func:`run_cli` (see :func:`make_parser_class`):

- parameter names are the canonical long flags and the keys of ``--config`` YAML files;
- ``COMMON_ALIASES`` gives the short flags shared by all tools (``-i``, ``-o``, ``-v``, ``-q``, ``-f``, ``-y``),
  per tool ``aliases`` add the specific ones, including legacy long flags kept for compatibility;
- every ``snake_case`` flag is also accepted with dashes (``--image-transfer``);
- ``bool`` parameters are flags: ``--force`` / ``--no_force``, no value needed;
- ``--verbose`` accepts an optional value: bare ``-v`` means ``info``;
- ``--config file.yaml`` and ``--print_config`` come from jsonargparse.
"""

import logging
import typing
from typing import Callable, Dict, List, Optional, Union

from docstring_parser import DocstringStyle
from jsonargparse import CLI, ActionYesNo, ArgumentParser, set_parsing_settings

Verbosity = Union[int, str]
""" A logging level given by name (debug, info, warning, ...) or by integer value. """

DEFAULT_VERBOSITY: Verbosity = 'warning'

COMMON_ALIASES: Dict[str, List[str]] = {
    '--verbose': ['-v'],
    '--quiet': ['-q', '--silent'],
    '--force': ['-f', '-y'],
    '--input': ['-i'],
    '--inputs': ['-i'],
    '--output': ['-o'],
}


_active_aliases: Dict[str, List[str]] = {}
""" alias table of the parser being built, see make_parser_class """


def _list_item_type(type_hint) -> Optional[type]:
    """ List[X] or Optional[List[X]] -> X, anything else -> None """
    if typing.get_origin(type_hint) is Union:
        not_none = [t for t in typing.get_args(type_hint) if t is not type(None)]
        type_hint = not_none[0] if len(not_none) == 1 else None
    if typing.get_origin(type_hint) in (list, List):
        args = typing.get_args(type_hint)
        return args[0] if args else str
    return None


def make_parser_class(aliases: Dict[str, List[str]]) -> type:
    """
    Build an ArgumentParser class applying the kapture command line conventions.
    This is the "parser_class" workaround for short aliases, see https://github.com/mauvilsa/jsonargparse/issues/840

    :param aliases: extra option strings per canonical long flag, e.g. {'--output': ['-k', '--kapture']}
    :return: an ArgumentParser subclass
    """
    # NB: the alias table is kept in a module-level variable, not in a closure: for class based tools (subcommands)
    # jsonargparse re-compiles add_argument into its argument group classes, and closures are lost there.
    _active_aliases.clear()
    _active_aliases.update({flag: COMMON_ALIASES.get(flag, []) + aliases.get(flag, [])
                            for flag in list(COMMON_ALIASES) + [a for a in aliases if a not in COMMON_ALIASES]})

    class KaptureArgumentParser(ArgumentParser):
        def add_argument(self, *args, **kwargs):
            if args and isinstance(args[0], str) and args[0].startswith('--'):
                long_flag = args[0]  # jsonargparse may pass more option strings, eg. ('--inputs', '--inputs+')
                # also accept the dashed form of snake_case flags
                if '_' in long_flag:
                    args += (long_flag.replace('_', '-'),)
                # short (and legacy long) aliases
                args += tuple(a for a in _active_aliases.get(long_flag, ()) if a not in args)
                # jsonargparse hides the type hint of signature parameters inside its action
                type_hint = getattr(kwargs.get('action'), '_typehint', None) or kwargs.get('type')
                # booleans are flags: --force / --no_force
                if type_hint is bool:
                    kwargs.pop('type', None)
                    kwargs['action'] = ActionYesNo
                # lists take space separated values: --inputs a b c (as argparse nargs='+')
                item_type = _list_item_type(type_hint)
                if item_type is not None:
                    kwargs.pop('action', None)
                    kwargs['type'] = item_type
                    kwargs['nargs'] = '+'
                    if typing.get_origin(item_type) is typing.Literal:  # keep the help readable
                        kwargs['metavar'] = 'ITEM'
                        kwargs['help'] = (kwargs.get('help') or '') + ' choose among: ' + ', '.join(
                            str(c) for c in typing.get_args(item_type))
                # bare -v means info
                if long_flag == '--verbose':
                    kwargs.update(nargs='?', const='info')
            return super().add_argument(*args, **kwargs)

    return KaptureArgumentParser


def run_cli(tool: Callable, aliases: Dict[str, List[str]] = {}, **cli_kwargs):
    """
    Run the given function as a command line tool.

    :param tool: the type-hinted function implementing the tool.
    :param aliases: extra option strings per canonical long flag, e.g. {'--output': ['-k', '--kapture']}
    :param cli_kwargs: forwarded to jsonargparse.CLI (eg. args=[...] for tests).
    :return: what the tool returns.
    """
    set_parsing_settings(docstring_parse_style=DocstringStyle.REST)
    return CLI(tool, parser_class=make_parser_class(aliases), as_positional=False, **cli_kwargs)


def to_logging_level(verbose: Verbosity) -> int:
    """
    Convert a verbosity (name or integer, as string or not) to a logging level.

    :param verbose: 'debug', 'INFO', 10, '10', ...
    :return: logging level, eg. logging.DEBUG
    """
    try:
        return int(verbose)
    except ValueError:
        level = logging.getLevelName(str(verbose).upper())
        if not isinstance(level, int):
            raise ValueError(f'unknown verbosity level "{verbose}"')
        return level


def setup_logging(logger: logging.Logger,
                  verbose: Verbosity = DEFAULT_VERBOSITY,
                  quiet: bool = False) -> int:
    """
    Set the logging level of the tool logger, and of the kapture library logger when debugging.

    :param logger: the tool logger.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: if True, silence everything but critical messages.
    :return: the logging level applied.
    """
    level = logging.CRITICAL if quiet else to_logging_level(verbose)
    logger.setLevel(level)
    if level <= logging.DEBUG:
        # also let kapture express its logs
        logging.getLogger('kapture').setLevel(level)
    return level
