"""UNIONS v2.0.0: from the calibrated catalogue to a blinded pseudo-Cℓ of bin 4.

Run inside the sp_validation container (see run_demo.sh).
"""
import ssl
from pathlib import Path

import aiohttp
import lsdb
import pymaster as nmt
from upath import UPath

from sp_validation import pseudo_cl, sacc_io
from sp_validation.blinding import Blind

ROOT = "https://ws-cadc.canfar.net/vault/files/cfis/weak_lensing/ShapePipe/v2.x/v2.0.0"
CERT = str(Path.home() / ".ssl/cadcproxy.pem")
BIN = 4  # TOM_BIN_ID 1–4; SACC tracer source_3


async def cadc_client(loop=None, certfile=None, **kwargs):
    context = ssl.create_default_context()
    context.load_cert_chain(certfile)
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=context), **kwargs)


def vault(path):
    return UPath(f"{ROOT}/{path}", get_client=cadc_client, client_kwargs={"certfile": CERT})


# 1. The columns a shear power spectrum needs, for one bin, saved locally.
columns = ["RA", "Dec", "e1", "e2", "w_des", "TOM_BIN_ID"]
gal = lsdb.open_catalog(vault("catalogs/calibrated"), columns=columns)
gal = gal.query(f"TOM_BIN_ID == {BIN}").compute()
gal.to_parquet(f"bin{BIN}.parquet")
ra, dec, e1, e2, w = (gal[c].to_numpy() for c in columns[:5])

# 2. Pseudo-Cℓ with NaMaster on the catalogue (no map), cosmo_val's binning.
b = pseudo_cl.make_namaster_bin(*pseudo_cl.pseudo_cl_geometry(1024), "powspace", n_ell_bins=32)
f, _, wsp = pseudo_cl.get_field_and_workspace_from_catalog(b, ra, dec, e1, e2, w)
ee, eb, _, bb = wsp.decouple_cell(nmt.compute_coupled_cell(f, f))
window_ells, window = pseudo_cl.bandpower_window_from_workspace(wsp)

# 3. One SACC with the bin's n(z); saving it under the blind conceals Cℓ_EE.
z, nz = sacc_io.read_nz("nz_tomo.txt")
s = sacc_io.new_sacc({BIN - 1: (z, nz[:, BIN - 1])})
pair = (BIN - 1, BIN - 1)
sacc_io.add_pseudo_cl(s, pair, b.get_effective_ells(), ee, bb, eb, window_ells=window_ells, window_weights=window)
sacc_io.save(s, f"bin{BIN}_blinded.sacc", blind=Blind("unions", Path("unions.blind.json")))
