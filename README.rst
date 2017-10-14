========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires| |coveralls| |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |commits-since|

.. |docs| image:: https://readthedocs.org/projects/python-mics/badge/?style=flat
    :target: https://readthedocs.org/projects/python-mics
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/craabreu/python-mics.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/craabreu/python-mics

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/craabreu/python-mics?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/craabreu/python-mics

.. |requires| image:: https://requires.io/github/craabreu/python-mics/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/craabreu/python-mics/requirements/?branch=master

.. |coveralls| image:: https://coveralls.io/repos/craabreu/python-mics/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/craabreu/python-mics

.. |codecov| image:: https://codecov.io/github/craabreu/python-mics/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/craabreu/python-mics

.. |version| image:: https://img.shields.io/pypi/v/mics.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/mics

.. |commits-since| image:: https://img.shields.io/github/commits-since/craabreu/python-mics/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/craabreu/python-mics/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/mics.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/mics

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/mics.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/mics


.. end-badges

Mixtures of Independently Collected Samples

* Free software: MIT license

Installation
============

::

    pip install mics

Documentation
=============

https://python-mics.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
