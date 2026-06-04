Installation
============

``sp_validation`` is **not** distributed on PyPI. The recommended way to install
it is through a pre-built container; a development checkout with ``uv`` is the
alternative when you need to edit the source.

Container (recommended)
-----------------------

Container images carrying the full scientific stack are built automatically and
pushed to the `GitHub Container Registry (GHCR)
<https://github.com/CosmoStat/sp_validation/pkgs/container/sp_validation>`_ on
every push to ``develop``. The image runs on most systems, including HPC
clusters, with no further setup.

We recommend `Apptainer <https://apptainer.org>`_ (formerly Singularity), which
is installed on most clusters:

.. code-block:: bash

   # Build a writeable "sandbox" container in the current directory.
   # ./sp_validation is a directory that behaves like a small VM.
   apptainer build --sandbox sp_validation docker://ghcr.io/cosmostat/sp_validation:develop

   # Open a shell in the container ...
   apptainer shell --writable sp_validation

   # ... and confirm the installation works.
   python -c "import sp_validation"

The image also runs under Docker:

.. code-block:: bash

   docker run --rm -it ghcr.io/cosmostat/sp_validation:develop python -c "import sp_validation"

.. note::

   We do not currently build images for Apple Silicon / arm64. The amd64 images
   run on these machines through emulation, with reduced performance.

Development install
-------------------

To work on the source — running the test suite or building these docs — clone
the repository and install it in editable mode. The package is managed with
`uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/CosmoStat/sp_validation.git
   cd sp_validation
   uv pip install -e '.[develop]'

The ``develop`` extra pulls in the testing and documentation dependencies. For a
narrower install, ``.[test]`` adds only the test extras and ``.[docs]`` only the
Sphinx stack.

.. note::

   ``sp_validation`` requires Python 3.11 or newer and depends on a large
   scientific stack (``treecorr``, ``pyccl``, ``healpy``, ``pymaster``, and
   others). A bare development install builds these from source, which can be
   slow and platform-sensitive; the container is the more reliable path for most
   users.
