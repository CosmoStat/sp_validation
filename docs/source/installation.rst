Installation
============

``sp_validation`` is **not** distributed on PyPI.
It runs from a pre-built container, managed by the bundled ``spv-container`` CLI; check out the source with ``uv`` only when you need to edit the package itself.

Container via ``spv-container`` (recommended)
---------------------------------------------

Every push builds an image carrying the full scientific stack and pushes it to the `GitHub Container Registry (GHCR)
<https://github.com/CosmoStat/sp_validation/pkgs/container/sp_validation>`_, tagged by branch — ``:develop`` tracks the integration branch.
The image runs on most systems, including HPC clusters, with no further setup.
``spv-container`` installs your personal copy of it and manages it from then on:

.. code-block:: bash

   git clone https://github.com/CosmoStat/sp_validation.git
   cd sp_validation
   ln -s "$PWD/src/sp_validation/container.py" ~/.local/bin/spv-container

   spv-container pull                                    # fetch the image (~1.5 GB)
   spv-container exec python -c "import sp_validation"   # confirm it works

The symlink works because ``container.py`` is deliberately stdlib-only: it runs on the *host*, where the science stack is not installed.
(Inside the container the same CLI is on ``PATH`` as a console script.)
``pull`` requires `Apptainer <https://apptainer.org>`_ (formerly Singularity), which is installed on most clusters, and writes the image to one canonical per-user path, ``~/.cache/sp_validation/sp_validation.sif``.
Each user owns their copy: you refresh it when you want to, and nobody else's refresh moves the ground under your running jobs.
On a cluster, run the pull from a compute node — it moves ~1.5 GB.

The subcommands:

.. code-block:: bash

   spv-container pull                     # fetch the published image to the canonical path
   spv-container status                   # what is here, which commit built it, how current
   spv-container exec <cmd...>            # run a command inside it (exec bash for a shell)
   spv-container sandbox                  # unpack into a writable dir, for pip installs
   spv-container exec --writable <cmd...> # ... with writes that persist

``status`` compares the image's build commit against your checkout's ``HEAD``, so you always know whether a ``pull`` would refresh anything.
The **sandbox** is the escape hatch for exploratory work that needs a package the image does not carry yet: once built, it takes precedence over the SIF everywhere — Snakemake workflow jobs included — until you reset with ``spv-container pull`` + ``spv-container sandbox --force``.

Everything resolves the image in one order — sandbox if it exists, else your SIF, else the registry tag — and that includes the analysis workflow.
How the workflow uses the image (Snakemake runs on the host; the profile puts each job in the container) is covered in ``workflow/README.md``.

Other ways to run the image
---------------------------

The published image is a normal OCI image; ``spv-container`` is a convenience, not a gatekeeper.
Run it directly with Apptainer:

.. code-block:: bash

   apptainer pull sp_validation.sif docker://ghcr.io/cosmostat/sp_validation:develop
   apptainer shell sp_validation.sif

or with Docker:

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
