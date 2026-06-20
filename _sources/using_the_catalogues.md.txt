# Using the weak-lensing catalogues

This tutorial shows how to *use* a public UNIONS/CFIS ShapePipe weak-lensing
catalogue from a downstream analysis: how to open it, inspect its contents,
measure a shear–shear correlation function, and — for the extended catalogue —
how to apply the metacalibration corrections yourself.

> **Author:** Martin Kilbinger \<martin.kilbinger@cea.fr\>
>
> The ShapePipe core team: Axel Guinot, Sam Farrens, Tobias Liaudat.
>
> ShapePipe is publicly available at <https://github.com/CosmoStat/shapepipe>.
> See [arXiv:2204.04798](https://arxiv.org/abs/2204.04798) for the analysis of
> an earlier version of the catalogue.

```{note}
The examples below target catalogue **v1.0** (April 2022), which is distributed
as FITS. From ShapePipe catalogue **v1.4.1** onward the merged catalogues ship
as HDF5 instead; open those with {func}`sp_validation.io.read_hdf5_file` (or
`h5py` / `astropy`) in place of `astropy.io.fits` below — the column names and
the calibration recipe are unchanged.
```

Two catalogues are released for each version, and they are used very
differently:

- a **base** catalogue, already calibrated and meant to be used *as is*;
- an **extended** catalogue, which carries the metacalibration information and
  must be *calibrated before use* (and may be cut beforehand).

```python
# Required python libraries and settings
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt

plt.rcParams['font.size'] = 20

# Optional: to compute the example shear–shear correlation functions
import treecorr

# Catalogue name base and version
cat_base = 'unions_shapepipe'
year = 2022
version = 'v1.0'
```

## 1. Base calibrated catalogue

This catalogue should be used **as is**. The shear estimates are calibrated for
the entire catalogue, so:

- do **not** match it to another sample (e.g. spectroscopic galaxies, or a
  LensFit catalogue);
- do **not** apply property-dependent cuts (e.g. in magnitude).

Selections that do *not* correlate with galaxy properties — spatial sub-areas,
regions around clusters, and so on — are fine. If you need property-dependent
cuts, recalibrate using the extended catalogue (Section 2).

### Open the file and inspect it

```python
cat_name = f'{cat_base}_{year}_{version}.fits'
print(f'Opening base catalogue {cat_name}...')
hdu_list = fits.open(cat_name)
data = hdu_list[1].data

data.dtype.names   # column names
len(data)          # number of objects
hdu_list[0].header # primary header (global calibration quantities)
hdu_list[1].header # table header (column units)
```

### Example 1: magnitude distribution

```python
plt.hist(data['mag'], bins=100, range=(19, 26))
plt.xlabel('$r$-band magnitude')
_ = plt.ylabel('frequency')
```

### Example 2: shear–shear correlation function

Build a TreeCorr configuration, taking the coordinate units straight from the
FITS header so RA and Dec are interpreted correctly:

```python
coord_unit = hdu_list[1].header['TUNIT1']
if coord_unit != hdu_list[1].header['TUNIT2']:
    raise ValueError('Units for RA and Dec should be equal')

sep_unit = 'arcmin'
TreeCorrConfig = {
    'ra_units': coord_unit,
    'dec_units': coord_unit,
    'min_sep': '1',
    'max_sep': '200',
    'sep_units': sep_unit,
    'nbins': 20,
}
```

Create the catalogue and correlation objects, then process (this can take a few
minutes):

```python
cat_gal = treecorr.Catalog(
    ra=data['RA'],
    dec=data['Dec'],
    g1=data['e1'],
    g2=data['e2'],
    w=data['w'],
    ra_units=coord_unit,
    dec_units=coord_unit,
)
gg = treecorr.GGCorrelation(TreeCorrConfig)
gg.process(cat_gal, cat_gal)
```

```python
plt.figure(figsize=(14, 8))
plt.errorbar(gg.meanr, gg.xip, yerr=np.sqrt(gg.varxip), label=r'$\xi_+$')
plt.errorbar(gg.meanr, gg.xim, yerr=np.sqrt(gg.varxim), label=r'$\xi_-$')
plt.loglog()
plt.legend()
plt.xlabel(rf'$\theta$ [{sep_unit}]')
_ = plt.ylabel(r'$\xi_\pm(\theta)$')
```

## 2. Extended catalogue

This catalogue must be **calibrated before use**. It carries the information
needed for calibration with the method of *metacalibration*
([arXiv:1702.02600](https://arxiv.org/abs/1702.02600)). Cuts may be applied
*before* calibration.

```{tip}
The `sp_validation` library performs every step below for you:
{func}`sp_validation.calibration.get_calibrated_m_c` calibrates an in-memory
catalogue for multiplicative and additive bias, and
{func}`sp_validation.calibration.get_calibrate_e_from_cat` does the same from a
catalogue on disk. The worked example here is kept for understanding what those
functions compute.
```

### Open the file

```python
cat_ext_name = f'{cat_base}_extended_{year}_{version}.fits'
print(f'Opening catalogue {cat_ext_name}...')
hdu_list_ext = fits.open(cat_ext_name)
data_ext = hdu_list_ext[1].data

data_ext.dtype.names  # column names
len(data_ext)         # number of objects
```

### Step 1 — Select the (sub-)sample

The calibration depends on the properties of the galaxy sample, so it must be
carried out *after* selecting the sample. The dummy example below keeps
everything; uncomment a line to apply a real selection.

```python
mask = np.full(len(data_ext), True)

# Other examples:
# mask = data_ext['mask_extern'] == 0  # LensFit-unmasked regions
# mask = data_ext['patch'] == 3        # patch P3
# mask = data_ext['mag'] < 23.5        # r-band magnitude cut

n_kept, n_all = np.count_nonzero(mask), len(data_ext)
print(f'Keeping {n_kept}/{n_all} = {n_kept / n_all * 100:.1f}% objects')
data_ext_sub = data_ext[mask]
```

### Step 2 — Compute the biases

**Additive bias.** For a survey as large as UNIONS/CFIS the mean shear over the
observed area is an excellent approximation to zero, so the additive bias is the
weighted mean of the uncalibrated ellipticities (also stored in the FITS header
for the full sample):

```python
c = np.empty(2)
for comp in (0, 1):
    c[comp] = np.average(data_ext_sub[f'e{comp+1}_uncal'], weights=data_ext_sub['w'])

print('Additive bias')
for comp in (0, 1):
    print(f'c_{comp+1} = {c[comp]:.3g}')
```

**Multiplicative bias.** In metacalibration this is a 2×2 matrix
$R = R_g + R_\mathrm{s}$, the sum of the shear response $R_g$ and the selection
response $R_\mathrm{s}$. Both are also stored in the header for the full sample.

The per-galaxy shear response is part of the extended catalogue, so the
ensemble $R_g$ is computed *after* the sub-sample selection:

```python
R_g = np.empty((2, 2))
for idx in (0, 1):
    for jdx in (0, 1):
        R_g[idx, jdx] = np.mean(data_ext_sub[f'R_g{idx+1}{jdx+1}'])

print('Shear response matrix R_g =')
print(np.matrix(R_g))
```

The selection response was pre-computed for the *entire* sample and cannot be
obtained per galaxy, so this catalogue version reuses the global
$R_\mathrm{s}$ for sub-samples too. Since $|R_\mathrm{s}| < |R_g|$ the resulting
error is small.

```python
R_s = np.empty((2, 2))
for idx in (0, 1):
    for jdx in (0, 1):
        R_s[idx][jdx] = hdu_list_ext[0].header[f'R_S{idx+1}{jdx+1}']

print('Selection response matrix R_s =')
print(np.matrix(R_s))

R = R_g + R_s
print('Total response matrix R =')
print(np.matrix(R))
```

### Step 3 — Apply the calibration

The observed and true shear are related by $\gamma^\mathrm{obs} = R\,\gamma + c$.
Subtract the additive term and multiply by the inverse total response:

```python
e_uncal_minus_c = np.array([
    data_ext['e1_uncal'] - c[0],
    data_ext['e2_uncal'] - c[1],
])
e_cal = np.linalg.inv(R).dot(e_uncal_minus_c)
```

As a cross-check (valid *only* when no cut was applied), the first calibrated
shears computed by hand should match those in the base catalogue:

```python
print(e_cal[:, 0:3])
print(data['e1'][0:3], data['e2'][0:3])
```
