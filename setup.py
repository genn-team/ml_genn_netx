from setuptools import setup, find_packages

setup(
    name="ml_genn_netx",
    version="0.0.2",
    packages=find_packages(),

    python_requires=">=3.7.0",
    install_requires = [
        "ml_genn>=2.3.0", "h5py"])
