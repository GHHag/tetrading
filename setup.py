from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'TETrading trading system development software'
LONG_DESCRIPTION = DESCRIPTION

setup(
    name='TETrading',
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='GHHag',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['pandas', 'numpy', 'matplotlib', 'mplfinance']
)
