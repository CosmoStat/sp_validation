Installation
============

``sp_validation`` is **not** distributed on PyPI.
Install it from a pre-built container, or check out the source with ``uv`` when you need to edit it.

Container (recommended)
-----------------------

Every push to ``develop`` builds an image carrying the full scientific stack and pushes it to the `GitHub Container Registry (GHCR)
<https://github.com/CosmoStat/sp_validation/pkgs/container/sp_validation>`_.
The image runs on most systems, including HPC clusters, with no further setup.

`Apptainer <https://apptainer.org>`_ (formerly Singularity) is installed on most clusters and is the path we recommend:

.. code-block:: bash

   # Build a writeable "sandbox" container in the current directory.
   # ./sp_validation is a directory that behaves like a small VM.
   apptainer build --sandbox sp_validation docker://ghcr.io/cosmostat/sp_validation:develop

   # Open a shell in the container, then confirm the install works.
   apptainer shell --writable sp_validation
   python -c "import sp_validation"

The image also runs under Docker:

.. code-block:: bash

   docker run --rm -it ghcr.io/cosmostat/sp_validation:develop python -c "import sp_validation"

.. note::

   We do not build images for Apple Silicon / arm64.
   The amd64 images run on these machines under emulation, more slowly.

Development install
-------------------

To run the test suite or build these docs, clone the repository and install it in editable mode.
The package is managed with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/CosmoStat/sp_validation.git
   cd sp_validation
   uv pip install -e '.[develop]'

The ``develop`` extra pulls in both the testing and documentation dependencies.
For a narrower install, ``.[test]`` adds only the test extras and ``.[docs]`` only the Sphinx stack.

.. note::

   ``sp_validation`` requires Python 3.11 or newer and pulls in a large scientific stack: ``treecorr``, ``pyccl``, ``healpy``, ``pymaster``, and others.
   A bare development install builds these from source, which is slow and platform-sensitive.
   For most users the container is the more reliable path.
