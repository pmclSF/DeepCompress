from setuptools import setup, find_namespace_packages

setup(
    name="deepcompress",
    version="2.0.0",
    package_dir={"": "src"},
    packages=find_namespace_packages(include=["*"], where="src"),
    python_requires=">=3.10",
    install_requires=[
        'numpy',
        'tensorflow>=2.11',
        'tensorflow-probability~=0.19',
        'matplotlib',
        'pandas',
        'tqdm',
        'pyyaml',
        'scipy',
        'numba',
        'keras-tuner',
    ],
)
