from setuptools import setup, find_packages


setup(
    packages=find_packages(),
    name="carla_gym_env",
    version="0.0.1",
    install_requires=[
        "episode_manager @ git+https://github.com/Haavasma/episode_manager.git",
        "gym",
    ],
)
