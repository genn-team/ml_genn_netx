from setuptools import setup, find_packages

setup(
    name="ml_genn_netx",
    version="0.0.1",
    packages=find_packages(),

    python_requires=">=3.7.0",
    install_requires = [
        "ml_genn>=2.1.0", "h5py"])
