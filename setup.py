from setuptools import setup, find_namespace_packages

setup(
    name="deepcompress",
    version="0.1",
    package_dir={"": "src"},
    packages=find_namespace_packages(include=["*"], where="src"),
    install_requires=[
        'numpy',
        'pytest',
        'numba'
    ],
)