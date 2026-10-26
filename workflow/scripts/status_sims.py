#!/usr/bin/env python3
"""Disk-scan status report for image-sims ShapePipe campaigns.

Not a DAG node (same reasoning as shapepipe/workflow/scripts/run_report.py:
a report that's part of the DAG would itself be poisoned by the failures it
must enumerate). Run any time, mid-run or after:

    workflow/bin/spv status -c sp_1p2z_grid_3.yaml    # one campaign (sp run config)
    workflow/bin/spv status <grid dir>                 # every sp_*.yaml in it
    python workflow/scripts/status_sims.py <run_dir>   # directly

For each tile/exposure and each stage, a manifest means complete, a log
with no manifest means failed, and neither means not yet attempted -- the
same three-way read completeness.py's own contract establishes.

A failed stage's log also carries why it failed: the runner counts, and the
lines completeness_sims.py scraped out of the shapepipe logs. Those are
reported per unit and grouped by reason, so one bad input among hundreds of
tiles is readable without opening any log by hand.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Two layouts share this script. The NATIVE sp_validation prototype fits the
# PSF at tile level (tile_fake_psf) and runs ngmix as one call; the UNIFIED
# ShapePipe workflow (shapepipe #891) moves the PSF to the exposure level
# (exp_psf), chunks ngmix into tile_ngmix_1..N and merges the chunks
# (tile_merge_cats). Pass --layout to force one; `auto` reads what is on disk.
NATIVE_TILE_STAGES = [
    "tile_get_images", "tile_uncompress", "tile_find_exposures",
    "tile_merge_headers", "tile_detect", "tile_fake_psf", "tile_vignets",
    "tile_ngmix", "tile_make_cat",
]
NATIVE_EXP_STAGES = ["exp_get_images", "exp_split"]

UNIFIED_TILE_STAGES = [
    "tile_get_images", "tile_uncompress", "tile_find_exposures",
    "tile_merge_headers", "tile_detect", "tile_vignets",
    "tile_ngmix", "tile_merge_cats", "tile_make_cat",
]
UNIFIED_EXP_STAGES = ["exp_get_images", "exp_split", "exp_psf"]

# Back-compat for anything importing these names.
TILE_STAGES = NATIVE_TILE_STAGES
EXP_STAGES = NATIVE_EXP_STAGES

# The Gen-2 RUN_NAME suffixes (run_sp_tile_Git, run_sp_exp_Gie, ...) --
# shorter and more familiar than the stage names for a wide table.
STAGE_ABBR = {
    "tile_get_images": "Git",
    "tile_uncompress": "Uz",
    "tile_find_exposures": "Fe",
    "tile_merge_headers": "Mh_exp",
    "tile_detect": "Sx",
    "tile_fake_psf": "fpsf",
    "tile_vignets": "ViVi",
    "tile_ngmix": "Ng",
    "tile_merge_cats": "Ms",
    "tile_make_cat": "Mc",
    "exp_get_images": "Gie",
    "exp_split": "Sp",
    "exp_psf": "Psf",
}

SYMBOL = {"complete": ".", "failed": "X", "missing": " "}


_CHUNK_RE = re.compile(r"^(tile_ngmix)_\d+$")


def unit_records(unit_dir: Path):
    """(done, failed) stage-name sets for one unit.

    A reclaimed unit has had its manifests deleted, so clean_tile/
    clean_exposure's tombstone is read as the record of what completed -- it
    absorbs every manifest verbatim for exactly this reason. Without this a
    FINISHED campaign reports all zeros, because the more complete a unit is
    the fewer manifests survive on disk.
    """
    done, failed = set(), set()
    mdir, ldir = unit_dir / "manifests", unit_dir / "logs"
    if mdir.is_dir():
        done |= {p.stem for p in mdir.glob("*.json")}
    if ldir.is_dir():
        # An EMPTY log means the job never ran, not that it failed: snakemake
        # creates the declared log path even for a job its cone never reached.
        # Counting those as failures turned one dead tile into a column of Xs
        # across every downstream stage (run_report calls them "not run").
        failed |= {p.stem for p in ldir.glob("*.json")
                   if p.stat().st_size > 0}
    tomb = unit_dir / "cleaned.json"
    if tomb.exists():
        try:
            body = json.loads(tomb.read_text())
        except (OSError, ValueError):
            body = {}
        done |= set((body.get("manifests") or {}))
    return done, failed - done


def unit_statuses(unit_dir: Path, stages: list, n_chunks: int) -> dict:
    """stage -> complete | failed | missing, with the chunked ngmix collapsed
    into its single `tile_ngmix` column (complete only when every chunk is)."""
    done, failed = unit_records(unit_dir)

    chunks_done = {m for m in done if _CHUNK_RE.match(m)}
    chunks_failed = {m for m in failed if _CHUNK_RE.match(m)}

    out = {}
    for s in stages:
        if s == "tile_ngmix" and n_chunks > 1:
            if len(chunks_done) >= n_chunks:
                out[s] = "complete"
            elif chunks_failed or chunks_done:
                out[s] = "failed" if chunks_failed else "missing"
            else:
                out[s] = "missing"
        elif s in done:
            out[s] = "complete"
        elif s in failed:
            out[s] = "failed"
        else:
            out[s] = "missing"
    return out


def units_of(run_dir: Path, kind: str) -> list:
    root = run_dir / kind
    return sorted(p for p in root.glob("*/*") if p.is_dir()) if root.is_dir() else []


def ngmix_chunks(run_dir: Path) -> int:
    """How many chunks this run splits ngmix into: the most any one tile
    records. 1 means the native single-call layout."""
    best = 1
    for unit_dir in units_of(run_dir, "tiles"):
        done, failed = unit_records(unit_dir)
        n = len({m for m in (done | failed) if _CHUNK_RE.match(m)})
        best = max(best, n)
    return best


def detect_layout(run_dir: Path) -> str:
    """'unified' or 'native'.

    Structural markers first, because they exist from the first minute of a
    run: the native prototype keeps a plain JSON index inside the run dir,
    while bin/sp puts its snakemake state in a `<run_dir>-state` sibling and
    its index in a sqlite file under the products root. Stage-name sniffing is
    the fallback, and on its own it mis-reads a young unified run as native --
    none of tile_ngmix_*, tile_merge_cats or exp_psf has been reached yet.
    """
    if (run_dir / "index.json").is_file():
        return "native"
    if run_dir.with_name(run_dir.name + "-state").is_dir():
        return "unified"
    for name in ("product", "products"):
        if (run_dir.parent / name / "index" / "run_index.sqlite").is_file():
            return "unified"

    for kind in ("tiles", "exp"):
        for unit_dir in units_of(run_dir, kind):
            done, failed = unit_records(unit_dir)
            names = done | failed
            if any(_CHUNK_RE.match(m) for m in names) or "tile_merge_cats" in names \
                    or "exp_psf" in names:
                return "unified"
            if "tile_fake_psf" in names:
                return "native"
    return "native"


def stage_status(unit_dir: Path, stage: str) -> str:
    """Kept for callers that ask about one stage; prefer unit_statuses."""
    return unit_statuses(unit_dir, [stage], 1)[stage]


# Group reasons that differ only in tile id, path or count, so N tiles failing
# the same way collapse to one line.
# Two segments minimum, so a count like "0/3" is not mistaken for a path.
_PATH_RE = re.compile(r"(?:/[^/\s]+){2,}")
_NUM_RE = re.compile(r"\d+")


def reason_shape(reason: str) -> str:
    return _NUM_RE.sub("N", _PATH_RE.sub("<path>", reason))


def short_paths(reason: str) -> str:
    """Absolute paths down to their file name: every run dir prefix is the
    same, and the full lines are far too wide to scan."""
    return _PATH_RE.sub(lambda m: m.group(0).rsplit("/", 1)[-1], reason)


def stage_failures(unit_dir: Path, stage: str, n_chunks: int = 1) -> list:
    """Why `stage` failed, as "<runner> found/expect" plus the scraped lines.

    Reads the log completeness_sims.py wrote. An unreadable or reason-less log
    still counts as a failure -- it just cannot say more than that.
    """
    logs = [unit_dir / "logs" / f"{stage}.json"]
    if stage == "tile_ngmix" and n_chunks > 1:
        # No tile_ngmix.json exists in the chunked layout; each chunk logs
        # separately, and any of them can be the one that failed.
        logs = sorted((unit_dir / "logs").glob("tile_ngmix_*.json"))

    out = []
    for log in logs:
        try:
            body = json.loads(log.read_text())
        except (OSError, ValueError):
            out.append(f"{log.name}: log unreadable")
            continue
        tag = f"{log.stem}: " if len(logs) > 1 else ""
        for rec in body.get("failures", []):
            head = f"{tag}{rec.get('runner')}: {rec.get('found')}/{rec.get('expect')}"
            reasons = rec.get("reasons") or []
            out.append(head if not reasons else f"{head} -- {reasons[0]}")
            out += reasons[1:]
    return out or ["no reason recorded"]


def report_failures(run_dir: Path, kind: str, stages: list,
                    n_chunks: int = 1) -> None:
    """Every failed unit with its reasons, then the same grouped by reason."""
    units = units_of(run_dir, kind)

    failed, groups = [], {}
    for unit_dir in units:
        statuses = unit_statuses(unit_dir, stages, n_chunks)
        for s in stages:
            if statuses[s] != "failed":
                continue
            reasons = stage_failures(unit_dir, s, n_chunks)
            failed.append((unit_dir.name, s, reasons))
            for r in reasons:
                key = (s, reason_shape(r))
                groups.setdefault(key, [r, []])[1].append(unit_dir.name)

    if not failed:
        print(f"no failed {kind}")
        print()
        return

    n_units = len({u for u, _, _ in failed})
    print(f"{n_units} {kind} failed, {len(failed)} stage failures")
    print()
    for unit, stage, reasons in failed:
        print(f"  {unit:<12} {STAGE_ABBR[stage]}")
        for r in reasons:
            print(f"      {short_paths(r)}")
    print()

    # Only worth a second pass when a reason actually spans several units.
    shared = {k: v for k, v in groups.items() if len(v[1]) > 1}
    if not shared:
        return
    print("  by reason:")
    for (stage, _), (reason, hit) in sorted(shared.items(),
                                            key=lambda kv: -len(kv[1][1])):
        print(f"    {len(hit):>4}x {STAGE_ABBR[stage]:<8} {short_paths(reason)}")
        print(f"         {', '.join(hit[:8])}" + (" ..." if len(hit) > 8 else ""))
    print()


def report_finished(run_dir: Path, kind: str, stages: list,
                    n_chunks: int = 1) -> None:
    """--very-short: the abbreviated header, and ONE row below it -- how many
    units have COMPLETED each step -- instead of one row per unit."""
    units = units_of(run_dir, kind)
    n_cleaned = sum(1 for u in units if (u / "cleaned.json").exists())
    print(f"{len(units)} {kind}")
    if not units:
        return
    if n_cleaned:
        # Say it outright. The counts below come from the tombstones once a
        # unit is reclaimed, so a FINISHED, fully cleaned campaign prints a row
        # of full numbers and looks untouched -- the opposite of the pre-
        # tombstone behaviour, where it printed zeros and looked emptied.
        what = "store deleted" if n_cleaned == len(units) else "stores deleted"
        print(f"  {n_cleaned}/{len(units)} RECLAIMED ({what}); the row below is "
              f"what ran, read back from cleaned.json")

    header = f"{'':<12}" + "".join(f"{STAGE_ABBR[s]:>8}" for s in stages)
    print(header)
    print("-" * len(header))

    counts = {s: 0 for s in stages}
    for unit_dir in units:
        statuses = unit_statuses(unit_dir, stages, n_chunks)
        for s in stages:
            if statuses[s] == "complete":
                counts[s] += 1
    row = f"{'complete':<12}" + "".join(f"{counts[s]:>8}" for s in stages)
    print(row)


def report_units(run_dir: Path, kind: str, stages: list, *, short: bool = False,
                 n_chunks: int = 1) -> None:
    units = units_of(run_dir, kind)
    if not units:
        print(f"No {kind} units found under {run_dir / kind}")
        return

    counts = {s: {"complete": 0, "failed": 0, "missing": 0} for s in stages}

    header = f"{'unit':<12}" + "".join(f"{STAGE_ABBR[s]:>8}" for s in stages)
    print(header)
    print("-" * len(header))

    for unit_dir in units:
        unit = unit_dir.name
        row = f"{unit:<12}"
        statuses = unit_statuses(unit_dir, stages, n_chunks)
        for s in stages:
            st = statuses[s]
            counts[s][st] += 1
            row += f"{SYMBOL[st]:>8}"
        print(row)

    # --short: just the per-unit table above, skip the aggregate counts below.
    if short:
        print()
        return

    print()
    print(f"{len(units)} {kind} total")
    for s in stages:
        c = counts[s]
        print(f"  {STAGE_ABBR[s]:<8} complete={c['complete']:<4} failed={c['failed']:<4} "
              f"not-yet-run={c['missing']}")
    print()


def final_cats(products_dir: Path) -> list:
    """The published catalogues, on the products root (which is run_dir when
    the run config sets no products_dir)."""
    return sorted(products_dir.glob("tiles/*/*/final_cat-*.fits"))


def guess_products(run_dir: Path) -> Path:
    """Where the catalogues are when only a run dir was given.

    The native layout publishes under the run root itself; the unified one
    puts run/ and product/ side by side, so a bare `status_sims.py <run_dir>`
    would otherwise report `final_cat produced: 0` for a finished campaign.
    """
    if final_cats(run_dir):
        return run_dir
    for name in ("product", "products"):
        cand = run_dir.parent / name
        if final_cats(cand):
            return cand
    return run_dir


def report_disk(run_dir: Path, enabled: bool) -> None:
    """What the run directory still occupies -- the direct answer to "was this
    cleaned up?", which the stage table alone cannot give."""
    if not enabled:
        return
    total = files = 0
    for dirpath, _, names in os.walk(run_dir):
        for n in names:
            try:
                total += os.stat(os.path.join(dirpath, n)).st_size
                files += 1
            except OSError:
                pass
    for unit, div in (("T", 1 << 40), ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if total >= div:
            size = f"{total / div:.1f}{unit}"
            break
    else:
        size = f"{total}B"
    print(f"run dir on disk: {size} in {files} files")


def report_merged(products_dir: Path, cats: list) -> None:
    """The merged hdf5, the campaign's hand-off to sp_validation.

    Per-tile catalogues are only half the story: the sp_validation side starts
    from ONE hdf5 per branch, so a campaign whose tiles are all green but whose
    merge never ran is not actually finished. Checked against the per-tile
    FITS, because a merge that silently picked up a subset is the failure worth
    catching -- create_final_cat.py skips a tile it cannot find and says so
    only in its own verbose output.
    """
    merged = sorted(products_dir.glob("final_cat_*.hdf5"))
    if not merged:
        print(f"merged hdf5: none in {products_dir} "
              f"(run merge_final_cats, or create_final_cat.py -I -c <run config>)")
        return

    for path in merged:
        try:
            import h5py
        except ImportError:
            print(f"merged hdf5: {path.name} ({path.stat().st_size / 1e6:.0f} MB)"
                  "  -- install h5py to check its contents")
            return
        try:
            with h5py.File(path, "r") as fh:
                groups = {}
                def walk(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        groups[name] = obj.shape[0]
                fh.visititems(walk)
        except OSError as exc:
            print(f"merged hdf5: {path.name} UNREADABLE ({exc})")
            continue

        n_tiles, n_rows = len(groups), sum(groups.values())
        # Row totals from the per-tile FITS headers only -- no pixel data read.
        src_rows = 0
        for c in cats:
            try:
                from astropy.io import fits
                with fits.open(c, memmap=True) as hdus:
                    src_rows += hdus[1].header["NAXIS2"]
            except Exception:
                src_rows = -1
                break

        size = path.stat().st_size / 1e6
        note = ""
        if len(cats) and n_tiles != len(cats):
            note = f"  MISMATCH: {len(cats)} final_cat files on disk"
        elif src_rows >= 0 and n_rows != src_rows:
            note = f"  MISMATCH: {src_rows:,} rows in the per-tile catalogues"
        print(f"merged hdf5: {path.name}  {n_tiles} tiles, {n_rows:,} rows, "
              f"{size:.0f} MB{note}")

    n_file = products_dir / "n_tiles_final.txt"
    if n_file.exists():
        recorded = n_file.read_text().strip()
        flag = "" if recorded == str(len(cats)) else f"  MISMATCH: {len(cats)} on disk"
        print(f"n_tiles_final.txt: {recorded}{flag}")


def _expand(config: dict) -> dict:
    """Expand $name and ${name} in every value, from the top-level scalar keys,
    repeatedly (as ShapePipe's run_config.py does for an sp run config)."""
    scalars = {k: v for k, v in config.items()
               if isinstance(v, (str, int, float)) and not isinstance(v, bool)}

    def sub(value, depth=0):
        if isinstance(value, str):
            out = re.sub(r"\$\{(\w+)\}|\$(\w+)",
                         lambda m: str(scalars.get(m.group(1) or m.group(2),
                                                   m.group(0))), value)
            return out if out == value or depth >= 5 else sub(out, depth + 1)
        if isinstance(value, dict):
            return {k: sub(v, depth) for k, v in value.items()}
        if isinstance(value, list):
            return [sub(v, depth) for v in value]
        return value

    return {k: sub(v) for k, v in config.items()}


def run_dirs_from_config(path: Path):
    """(run_dir, products_dir) from a run config's outputs: block; products_dir
    defaults to run_dir, as in ShapePipe's workflow."""
    import yaml

    config = _expand(yaml.safe_load(Path(path).read_text()) or {})
    outputs = config.get("outputs") or {}
    run_dir = outputs.get("run_dir")
    if not run_dir or "$" in str(run_dir):
        raise ValueError(f"{path}: outputs.run_dir is unset or unexpanded "
                         f"({run_dir!r})")
    return Path(run_dir), Path(outputs.get("products_dir") or run_dir)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, nargs="?",
                   help="run directory; omit when passing -c")
    p.add_argument("-c", "--config", type=Path,
                   help="ShapePipe run config (sp_*.yaml): read outputs.run_dir "
                        "and outputs.products_dir from it, $name/${name} expanded")
    p.add_argument("-s", "--short", action="store_true",
                   help="per-tile/per-exposure table only, no aggregate "
                        "'N units total' / per-stage count summary")
    p.add_argument("-S", "--very-short", action="store_true",
                   help="no per-unit table at all -- just 'N/M finished' "
                        "for tiles and exposures, once each (overrides -s)")
    p.add_argument("-p", "--products", type=Path,
                   help="products root, when it is neither the run dir nor a "
                        "product/ sibling of it")
    p.add_argument("--du", action="store_true",
                   help="also measure the run directory on disk (a full walk; "
                        "slow on a live campaign, instant on a reclaimed one)")
    p.add_argument("--layout", choices=("auto", "native", "unified"),
                   default="auto",
                   help="stage set to report: the sp_validation native "
                        "prototype, the unified ShapePipe workflow, or auto "
                        "(default) to read it off the run dir")
    args = p.parse_args(argv)

    if args.config:
        try:
            run_dir, products_dir = run_dirs_from_config(args.config)
        except (OSError, ValueError) as exc:
            p.error(f"{exc}")
    elif args.run_dir:
        run_dir = args.run_dir
        products_dir = guess_products(run_dir)
    else:
        p.error("give a run directory, or -c with a run config")

    if args.products:
        products_dir = args.products

    layout = args.layout if args.layout != "auto" else detect_layout(run_dir)
    tile_stages = UNIFIED_TILE_STAGES if layout == "unified" else NATIVE_TILE_STAGES
    exp_stages = UNIFIED_EXP_STAGES if layout == "unified" else NATIVE_EXP_STAGES
    n_chunks = ngmix_chunks(run_dir) if layout == "unified" else 1

    print(f"=== {run_dir} ===")
    if products_dir != run_dir:
        print(f"=== products: {products_dir} ===")
    chunk_note = f", ngmix x{n_chunks}" if n_chunks > 1 else ""
    print(f"=== layout: {layout}{chunk_note} ===")

    if args.very_short:
        report_finished(run_dir, "tiles", tile_stages, n_chunks)
        report_finished(run_dir, "exp", exp_stages)
        cats = final_cats(products_dir)
        print(f"final_cat produced: {len(cats)}")
        report_merged(products_dir, cats)
        report_disk(run_dir, args.du)
        return 0

    print()
    print("--- tiles ---")
    report_units(run_dir, "tiles", tile_stages, short=args.short, n_chunks=n_chunks)
    print("--- exposures ---")
    report_units(run_dir, "exp", exp_stages, short=args.short)

    if not args.short:
        print("--- failures ---")
        report_failures(run_dir, "tiles", tile_stages, n_chunks)
        report_failures(run_dir, "exp", exp_stages)

    report_disk(run_dir, args.du)
    cats = final_cats(products_dir)
    print(f"final_cat produced: {len(cats)}")
    if not args.short:
        for f in cats:
            print(f"  {f}")
    report_merged(products_dir, cats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
