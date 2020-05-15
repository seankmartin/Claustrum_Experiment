import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


DESCRIPTION = "bvmpc: Behavioural Med-PC analysis"
LONG_DESCRIPTION = read("README.md")

DISTNAME = 'bvmpc'
MAINTAINER = 'Sean Martin and Gao Xiang Ham'
MAINTAINER_EMAIL = 'martins7@tcd.ie'
URL = 'https://github.com/seankmartin/neuro-tools'
# TODO change to a release
DOWNLOAD_URL = 'https://github.com/seankmartin/neuro-tools'
VERSION = '0.1.0'

INSTALL_REQUIRES = [
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'scipy',
    'sklearn'
    # 'neo >= 0.8.0',
    # 'nixio >= 1.5.0b1'
]

PACKAGES = [
    'bvmpc'
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: Microsoft :: Windows',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":

    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          long_description_content_type="text/markdown",
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=INSTALL_REQUIRES,
          include_package_data=True,
          packages=PACKAGES,
          classifiers=CLASSIFIERS,
          )
