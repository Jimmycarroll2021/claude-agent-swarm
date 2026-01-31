"""
Setup script for claude-agent-swarm package.

This is a fallback for older pip versions.
Preferred method: pip install -e . or pip install .
"""

from setuptools import setup, find_packages

setup(
    name="claude-agent-swarm",
    version="1.0.0",
    description="A production-ready agent swarm framework for Claude Code",
    author="Claude Agent Swarm Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "anthropic>=0.42.0",
        "mcp>=1.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "structlog>=23.0.0",
        "aiosqlite>=0.19.0",
        "aiofiles>=23.0.0",
        "aiohttp>=3.9.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
        "redis": [
            "redis>=5.0.0",
            "aioredis>=2.0.0",
        ],
        "web": [
            "streamlit>=1.28.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "claude-swarm=claude_agent_swarm.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
