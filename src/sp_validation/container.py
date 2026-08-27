"""Manage this user's local copy of the sp_validation container image.

Everyone runs their own image. There is no shared image directory and no symlink
to keep honest: the canonical paths are under your own cache, you refresh them
when you want to, and nobody else's refresh moves the ground under a running job.

There are two layers, and you only need the second when you want it:

* the **SIF** (``~/.cache/sp_validation/sp_validation.sif``) -- a pristine,
  read-only copy of the published image. This is the default and the normal case.
* an optional **sandbox** (``~/.cache/sp_validation/sandbox/``) -- the same image
  unpacked into a writable directory, so ``pip install`` inside it sticks. This
  is the escape hatch for exploratory work that needs a package the image does
  not carry yet, and it is opt-in: nothing builds one for you.

Subcommands, exposed as the ``spv-container`` console script::

    spv-container pull                     # fetch the tag to the canonical path
    spv-container status                   # what is here, and how current is it
    spv-container sandbox                  # unpack the SIF into a writable dir
    spv-container exec <cmd...>            # run something inside it
    spv-container exec --writable <cmd...> # ... with writes that persist

Everything resolves the same image in the same order -- **sandbox if it exists,
else the SIF, else the registry tag** -- and that includes the Snakemake
workflow, so a package you installed into your sandbox is there for your
workflow jobs too.

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

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")) / "sp_validation"

# Where this user's image lives. Per-user by construction: one file, one owner,
# no coordination. Override with ``SPV_CONTAINER`` (an absolute path).
DEFAULT_SIF = CACHE_DIR / "sp_validation.sif"

# The optional writable unpacking of that image. Override with ``SPV_SANDBOX``.
DEFAULT_SANDBOX = CACHE_DIR / "sandbox"

# Bind mounts for interactive `exec`. candide's disks; override wholesale with
# ``SPV_APPTAINER_BINDS`` or per-call with ``--bind``.
DEFAULT_BINDS = "/home,/scratch,/automnt,/n17data,/n23data1,/n09data"


def local_sif():
    """Return this user's canonical image path (may not exist yet)."""
    override = os.environ.get("SPV_CONTAINER")
    path = Path(override) if override else DEFAULT_SIF
    return path.expanduser()


def local_sandbox():
    """Return this user's writable sandbox directory (may not exist)."""
    override = os.environ.get("SPV_SANDBOX")
    path = Path(override) if override else DEFAULT_SANDBOX
    return path.expanduser()


def resolve_image():
    """Return ``(path_or_uri, kind)`` for the image everything should run.

    The one resolution order, shared by the CLI and the workflow: the writable
    sandbox if it exists, else the pristine SIF if it exists, else the registry
    tag for Snakemake to pull. ``kind`` is ``"sandbox"``, ``"sif"`` or ``"tag"``.
    """
    sandbox = local_sandbox()
    if sandbox.is_dir():
        return str(sandbox), "sandbox"
    sif = local_sif()
    if sif.exists():
        return str(sif), "sif"
    return CONTAINER_URI, "tag"


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


def cmd_sandbox(args):
    """Unpack the image into a writable directory -- the opt-in escape hatch."""
    if shutil.which("apptainer") is None:
        sys.exit("apptainer is not on PATH")
    sandbox = local_sandbox()
    if sandbox.exists() and not args.force:
        sys.exit(
            f"sandbox already exists at {sandbox}\n"
            "pass --force to discard it and rebuild from a clean image"
        )
    source = args.source or (
        str(local_sif()) if local_sif().exists() else CONTAINER_URI
    )
    sandbox.parent.mkdir(parents=True, exist_ok=True)
    print(f"building sandbox from {source}\n     -> {sandbox}")
    # Build beside the target and swap it in, as `pull` does -- and for a sharper
    # reason here. A half-written .sif fails loudly, but a half-unpacked sandbox
    # *directory* is still a directory, so resolve_image() would elect it as the
    # live image and every job would silently run a broken tree.
    #
    # Building first also means a `--force` rebuild that fails (a typo in
    # --source, a network blip) leaves the sandbox you already had untouched,
    # rather than deleting a working environment on the way to not replacing it.
    #
    # `--fix-perms` so the tree can be deleted again later (apptainer warns about
    # exactly this otherwise). No `--fakeroot`: an unprivileged build from an
    # existing image works through user namespaces, which is what candide has.
    staging = sandbox.with_name(f"{sandbox.name}.build.{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        subprocess.run(
            ["apptainer", "build", "--sandbox", "--fix-perms", str(staging), source],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        sys.exit(f"sandbox build failed ({exc.returncode}); {sandbox} is unchanged")
    except (KeyboardInterrupt, OSError):
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if sandbox.exists():
        print(f"replacing {sandbox}")
        shutil.rmtree(sandbox, ignore_errors=True)
        if sandbox.exists():
            shutil.rmtree(staging, ignore_errors=True)
            sys.exit(f"could not remove {sandbox}; remove it by hand and retry")
    os.replace(staging, sandbox)
    print(
        "\nthis sandbox now takes precedence over the SIF everywhere, including "
        "workflow jobs.\ninstall into it with: spv-container exec --writable pip "
        "install <pkg>\nreset to a clean image with: spv-container pull && "
        "spv-container sandbox --force"
    )
    return 0


def cmd_status(args):
    """Report which image layer is live, its revision, and how current it is."""
    sif = local_sif()
    sandbox = local_sandbox()
    active, kind = resolve_image()

    if sif.exists():
        print(f"SIF:      {sif} ({sif.stat().st_size / 1e9:.1f} GB)")
    else:
        print(f"SIF:      absent ({sif})")
    if sandbox.is_dir():
        print(f"sandbox:  {sandbox} (writable; may carry local modifications)")
    else:
        print("sandbox:  none")

    if kind == "tag":
        print(f"\nactive:   {active} (registry tag -- nothing pulled locally)")
        print("run: spv-container pull")
        return 1

    print(f"\nactive:   {active} ({kind})")
    labels = image_labels(active)
    revision = labels.get("org.opencontainers.image.revision")
    source = ""
    if revision is None and kind == "sandbox" and sif.exists():
        # Some sandbox trees do not carry the original labels through. The SIF
        # beside it is the best remaining evidence of what it was built from --
        # a guess, so it is labelled as one rather than printed as fact.
        revision = image_revision(sif)
        if revision:
            source = " (inferred from the SIF beside it, not read from the sandbox)"
    print(f"revision: {revision or 'unknown'}{source}")
    print(f"version:  {labels.get('org.opencontainers.image.version', 'unknown')}")
    if kind == "sandbox":
        # The revision is the image the sandbox was *built from*; anything
        # installed into it since is invisible to any label. Say so rather than
        # let the revision read as a full description of what is running.
        print(
            "          (the revision above is what the sandbox was built from; "
            "anything\n           installed into it since is not reflected in "
            "any label)"
        )
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
    """Run a command inside the image -- the one-off path for humans and agents."""
    if shutil.which("apptainer") is None:
        sys.exit("apptainer is not on PATH")
    if not args.command:
        sys.exit("nothing to run; pass a command after `exec`")
    binds = args.bind or os.environ.get("SPV_APPTAINER_BINDS", DEFAULT_BINDS)

    if args.writable:
        # Writes only persist into a sandbox; a SIF is a read-only filesystem, so
        # `--writable` against one fails obscurely. Say what to do instead.
        sandbox = local_sandbox()
        if not sandbox.is_dir():
            sys.exit(
                f"--writable needs a sandbox, and there is none at {sandbox}\n"
                "build one with: spv-container sandbox"
            )
        image, extra = str(sandbox), ["--writable"]
    else:
        image, kind = resolve_image()
        if kind == "tag":
            sys.exit(f"no local image; run: spv-container pull ({image})")
        extra = []

    cmd = [
        "apptainer",
        "exec",
        *extra,
        "--cleanenv",
        "--bind",
        binds,
        image,
        *args.command,
    ]
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

    p_status = sub.add_parser(
        "status", help="report which image layer is live and how current it is"
    )
    p_status.set_defaults(func=cmd_status)

    p_sandbox = sub.add_parser(
        "sandbox", help="unpack the image into a writable directory (opt-in)"
    )
    p_sandbox.add_argument(
        "--source",
        help="image to unpack (default: the local SIF, or the registry tag)",
    )
    p_sandbox.add_argument(
        "--force",
        action="store_true",
        help="discard an existing sandbox and rebuild from a clean image",
    )
    p_sandbox.set_defaults(func=cmd_sandbox)

    p_exec = sub.add_parser("exec", help="run a command inside the local image")
    p_exec.add_argument("--bind", help=f"bind mounts (default: {DEFAULT_BINDS})")
    p_exec.add_argument(
        "--writable",
        action="store_true",
        help="run against the sandbox so writes (e.g. pip install) persist",
    )
    p_exec.add_argument("command", nargs=argparse.REMAINDER)
    p_exec.set_defaults(func=cmd_exec)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
