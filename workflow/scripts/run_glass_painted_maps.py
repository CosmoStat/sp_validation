"""Snakemake entry point for projected, catalogue-independent mock maps."""

from sp_validation.glass_mock import GlassMockConfig, generate_lensing_maps


def main():
    settings = dict(nside=4096, harmonic_lmax=8192, zmax=3.5, limber=True)
    settings.update(snakemake.params.map_config)
    settings.update(seed=int(snakemake.wildcards.seed), nz_path=str(snakemake.input.nz))
    generate_lensing_maps(
        GlassMockConfig(**settings),
        snakemake.params.output,
        snakemake.params.cache,
        workers=snakemake.threads,
    )


if __name__ == "__main__":
    main()
