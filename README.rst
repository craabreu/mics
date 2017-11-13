.. start-badges

.. list-table::
    :stub-columns: 0

    * - |travis| |appveyor| |requires| |coveralls| |codecov|
    * - |version| |wheel| |supported-versions| |commits-since|
    * - |docs|

.. |docs| image:: https://readthedocs.org/projects/mics/badge/?style=flat
    :target: https://readthedocs.org/projects/mics
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/craabreu/mics.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/craabreu/mics

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/craabreu/mics?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/craabreu/mics

.. |requires| image:: https://requires.io/github/craabreu/mics/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/craabreu/mics/requirements/?branch=master

.. |coveralls| image:: https://coveralls.io/repos/craabreu/mics/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/craabreu/mics

.. |codecov| image:: https://codecov.io/github/craabreu/mics/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/craabreu/mics

.. |version| image:: https://img.shields.io/pypi/v/mics.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/mics

.. |commits-since| image:: https://img.shields.io/github/commits-since/craabreu/mics/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/craabreu/mics/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/mics.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/mics

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/mics.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/mics

.. end-badges

========
Overview
========

Mixtures of Independently Collected Samples

* Free software: MIT license

Installation
============

::

    pip install mics

Documentation
=============

https://mics.readthedocs.io/

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
