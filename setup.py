from setuptools import setup, find_packages


setup(
    packages=[package for package in find_packages() if package.startswith("gym_env")],
    name="srunner_gym_env",
    version="0.0.1",
    install_requires=[
        "episode_manager @ git+https://github.com/Haavasma/episode_manager.git",
        "gym",
    ],
)
