"""Manage this user's local copy of the sp_validation container image.

Everyone runs their own image file. There is no shared image directory and no
symlink to keep honest: the canonical path is under your own cache
(``~/.cache/sp_validation/sp_validation.sif``), you refresh it when you want to,
and nobody else's refresh moves the ground under a running job.

Three subcommands, exposed as the ``spv-container`` console script::

    spv-container pull            # fetch the tag to the canonical path
    spv-container status          # is it there, and which commit is it?
    spv-container exec <cmd...>   # run something inside it

This module is deliberately **stdlib-only** (``argparse``/``subprocess``/
``pathlib``). It runs on the *host*, outside the container, where the science
stack is not installed -- so it must import without it. That also means it works
straight from a checkout with no install at all::

    python3 src/sp_validation/container.py pull
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# The image every entry point names. CI builds and pushes one per branch, tagged
# by the sanitized branch name, so ``:develop`` tracks the integration branch.
# ``workflow/common.py`` re-exports this as ``CONTAINER_URI``; it is written down
# here, once.
CONTAINER_URI = "docker://ghcr.io/cosmostat/sp_validation:develop"

# Where this user's image lives. Per-user by construction: one file, one owner,
# no coordination. Override with ``SPV_CONTAINER`` (an absolute path).
DEFAULT_SIF = (
    Path(os.environ.get("XDG_CACHE_HOME", "~/.cache"))
    / "sp_validation"
    / "sp_validation.sif"
)

# Bind mounts for interactive `exec`. candide's disks; override wholesale with
# ``SPV_APPTAINER_BINDS`` or per-call with ``--bind``.
DEFAULT_BINDS = "/home,/scratch,/automnt,/n17data,/n23data1,/n09data"


def local_sif():
    """Return this user's canonical image path (may not exist yet)."""
    override = os.environ.get("SPV_CONTAINER")
    path = Path(override) if override else DEFAULT_SIF
    return path.expanduser()


def image_labels(sif):
    """Return the image's OCI labels as a dict, or ``{}`` if unreadable.

    Never raises: a missing file, a missing ``apptainer``, or a corrupt image
    all mean "we don't know", which every caller here treats as non-fatal.
    """
    sif = Path(sif)
    if not sif.exists() or shutil.which("apptainer") is None:
        return {}
    try:
        out = subprocess.run(
            ["apptainer", "inspect", "--labels", str(sif)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    labels = {}
    for line in out.stdout.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            labels[key.strip()] = value.strip()
    return labels


def image_revision(sif):
    """Return the sp_validation commit the image was built from, or ``None``."""
    return image_labels(sif).get("org.opencontainers.image.revision")


def _git(*args, cwd=None):
    """Run a git command, returning stripped stdout or ``None`` on any failure."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=cwd, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def compare_revision(revision, repo=None):
    """Place an image revision relative to a checkout's HEAD.

    Returns one of ``"in-sync"``, ``"behind"`` (the image predates HEAD),
    ``"ahead"`` (HEAD predates the image), ``"diverged"``, or ``"unknown"``
    (no revision label, no git, or a commit this clone has never fetched).
    """
    if not revision:
        return "unknown"
    repo = repo or Path(__file__).resolve().parents[2]
    head = _git("rev-parse", "HEAD", cwd=repo)
    if head is None:
        return "unknown"
    if head == revision:
        return "in-sync"
    if _git("cat-file", "-e", f"{revision}^{{commit}}", cwd=repo) is None:
        return "unknown"
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, head],
        capture_output=True,
        cwd=repo,
    )
    if ancestor.returncode == 0:
        return "behind"
    reverse = subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, revision],
        capture_output=True,
        cwd=repo,
    )
    return "ahead" if reverse.returncode == 0 else "diverged"


def cmd_pull(args):
    """Pull ``--tag`` to the canonical path, atomically."""
    if shutil.which("apptainer") is None:
        sys.exit("apptainer is not on PATH")
    sif = local_sif()
    sif.parent.mkdir(parents=True, exist_ok=True)
    # Pull to a sibling temp name and rename. `mv` within one directory is an
    # atomic rename, so a job either gets the whole old image or the whole new
    # one; pulling in place would leave the file half-written for the ~15
    # minutes the pull takes, and anything starting in that window would fail.
    # Jobs already running hold the old inode open and finish against it.
    tmp = sif.with_name(sif.name + f".pull.{os.getpid()}")
    print(f"pulling {args.tag}\n     -> {sif}")
    try:
        subprocess.run(
            ["apptainer", "pull", "--force", "--name", str(tmp), args.tag], check=True
        )
        os.replace(tmp, sif)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(f"pull failed ({exc.returncode})")
    except KeyboardInterrupt:
        tmp.unlink(missing_ok=True)
        raise
    labels = image_labels(sif)
    print(f"revision: {labels.get('org.opencontainers.image.revision', 'unknown')}")
    print(f"version:  {labels.get('org.opencontainers.image.version', 'unknown')}")
    return 0


def cmd_status(args):
    """Report the image's presence, revision, and standing against the checkout."""
    sif = local_sif()
    if not sif.exists():
        print(f"no image at {sif}\nrun: spv-container pull")
        return 1
    size_gb = sif.stat().st_size / 1e9
    print(f"image:    {sif} ({size_gb:.1f} GB)")
    labels = image_labels(sif)
    revision = labels.get("org.opencontainers.image.revision")
    print(f"revision: {revision or 'unknown'}")
    print(f"version:  {labels.get('org.opencontainers.image.version', 'unknown')}")
    verdict = compare_revision(revision)
    explain = {
        "in-sync": "matches this checkout's HEAD",
        "behind": "older than this checkout's HEAD -- pull to refresh",
        "ahead": "newer than this checkout's HEAD",
        "diverged": "on a different branch from this checkout",
        "unknown": "cannot compare (no label, or a commit this clone lacks)",
    }[verdict]
    print(f"checkout: {verdict} ({explain})")
    return 0


def cmd_exec(args):
    """Run a command inside the canonical image -- the one-off testing path."""
    if shutil.which("apptainer") is None:
        sys.exit("apptainer is not on PATH")
    sif = local_sif()
    if not sif.exists():
        sys.exit(f"no image at {sif}; run: spv-container pull")
    if not args.command:
        sys.exit("nothing to run; pass a command after `exec`")
    binds = args.bind or os.environ.get("SPV_APPTAINER_BINDS", DEFAULT_BINDS)
    cmd = ["apptainer", "exec", "--cleanenv", "--bind", binds, str(sif), *args.command]
    return subprocess.run(cmd).returncode


def build_parser():
    parser = argparse.ArgumentParser(
        prog="spv-container", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_pull = sub.add_parser("pull", help="fetch the image to the canonical path")
    p_pull.add_argument(
        "--tag",
        default=CONTAINER_URI,
        help=f"image to pull (default: {CONTAINER_URI})",
    )
    p_pull.set_defaults(func=cmd_pull)

    p_status = sub.add_parser("status", help="report the local image and its revision")
    p_status.set_defaults(func=cmd_status)

    p_exec = sub.add_parser("exec", help="run a command inside the local image")
    p_exec.add_argument("--bind", help=f"bind mounts (default: {DEFAULT_BINDS})")
    p_exec.add_argument("command", nargs=argparse.REMAINDER)
    p_exec.set_defaults(func=cmd_exec)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
