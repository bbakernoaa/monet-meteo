"""
Command-line interface for monet-meteo.

This module provides a simple CLI for the monet-meteo package.
"""

import argparse
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    """
    Main entry point for the monet-meteo CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    if argv is None:
        argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        prog="monet-meteo",
        description="Monet Meteo - A comprehensive meteorological library for atmospheric sciences",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=get_version()),
    )
    
    # Add subcommands here as needed
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Example: info command
    info_parser = subparsers.add_parser("info", help="Show package information")
    info_parser.add_argument(
        "--dependencies", 
        action="store_true", 
        help="Show dependency information"
    )
    
    args = parser.parse_args(argv)
    
    if args.command == "info":
        print(f"Monet Meteo version: {get_version()}")
        if args.dependencies:
            print("Dependencies:")
            # In a real implementation, this would show actual dependencies
            print("  - numpy")
            print("  - pint")
            print("  - xarray")
            print("  - scipy")
            print("  - dask")
    else:
        # If no command is specified, show version info by default
        print(f"Monet Meteo version: {get_version()}")
        print("Run 'monet-meteo --help' for more information.")
    
    return 0


def get_version() -> str:
    """
    Get the current version of the monet-meteo package.
    
    Returns:
        The version string
    """
    try:
        from ._version import version
        return version
    except ImportError:
        # Fallback to the version in __init__.py
        try:
            from . import __version__
            return __version__
        except ImportError:
            return "unknown"


if __name__ == "__main__":
    sys.exit(main())