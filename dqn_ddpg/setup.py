"""
Setup script for DQN vs DDPG comparison project
"""

from setuptools import setup, find_packages

setup(
    name="dqn-ddpg-comparison",
    version="0.1.0",
    author="DQN-DDPG Research Team",
    description="Educational reinforcement learning project comparing DQN and DDPG algorithms",
    long_description=open("CLAUDE.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "gymnasium[classic_control]>=0.26.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
        "tensorboard>=2.8.0",
        "imageio>=2.15.0",
        "imageio-ffmpeg",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dqn-ddpg-train=scripts.experiments.run_experiment:main",
            "dqn-ddpg-compare=scripts.experiments.run_same_env_experiment:main",
            "dqn-ddpg-video=scripts.video.core.create_realtime_combined_videos:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
)