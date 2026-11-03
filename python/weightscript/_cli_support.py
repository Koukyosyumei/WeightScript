"""Dependency-free parser helpers and process status policy."""

import argparse
import sys

from .exceptions import MissingDependencyError


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_version(parser):
    from . import __version__

    parser.add_argument(
        "--version", action="version", version=f"weightscript {__version__}"
    )


def run_cli(operation):
    try:
        operation()
        return 0
    except KeyboardInterrupt:
        print("weightscript: interrupted", file=sys.stderr)
        return 130
    except (MissingDependencyError, ModuleNotFoundError) as error:
        from .exceptions import dependency_error

        print(f"weightscript: {dependency_error(error)}", file=sys.stderr)
        return 3
    except (ValueError, OSError) as error:
        print(f"weightscript: {error}", file=sys.stderr)
        return 1
