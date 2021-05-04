import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyphonebot_extra",
    version="0.0.1",
    author="Yoonyoung Cho and Maximilian Schommer",
    description="Non-core pyphonebot software such as visualization and simulation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vi-robotics/pyphonebot_extra",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pyphonebot_core @ git+ssh://git@github.com/vi-robotics/pyphonebot-core',
        'opencv-python',
        'pyqtgraph',
        'PyQt5',
        'cho_util',
        'networkx',
        'pyopengl',
        'argcomplete',
        'gym',
        'pybullet',
        'shapely'
    ],
    extras_require={
    },
    python_requires='>=3.6',
)