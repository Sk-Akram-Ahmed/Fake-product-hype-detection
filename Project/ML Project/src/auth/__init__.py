"""
Authentication module for Hype Detection System
"""

from .github_auth import github_auth, GitHubAuth

__all__ = ["github_auth", "GitHubAuth"]
