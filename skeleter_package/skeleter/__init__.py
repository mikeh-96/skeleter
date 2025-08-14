"""
Skeleter - Git-Integrated Template Generator with AWS Parameter Store

A command-line templating utility that processes templates in GitHub repositories,
creates pull requests with rendered configurations, and optionally auto-merges them.
"""

__version__ = "1.0.0"
__author__ = "Skeleter Team"
__description__ = "Git-integrated template generator with AWS Parameter Store integration"

from .main import Skeleter, main

__all__ = ["Skeleter", "main"]
