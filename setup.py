from setuptools import setup, find_packages

VERSION = '1.0.1'
DESCRIPTION = 'TETrading trading system development software'
LONG_DESCRIPTION = DESCRIPTION

setup(
    name='TETrading',
    version='1.0.1',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='GHHag',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['pandas', 'numpy', 'matplotlib', 'mplfinance']
)
