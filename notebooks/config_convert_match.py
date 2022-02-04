import os

input_base = ['main_set_up', 'match_stats', 'metacal_global', 'psf_leakage', 'write_cat']

inp_str = '.ipynb '.join(input_base)


cmd = f'jupyter nbconvert --to script {inp_str} --stdout > validation.py'

os.system(cmd)

# Run validation from command line via
# ipython validation.py

"""
sp_base = f"{os.environ['HOME']}/sp_validation"

for sc in ['cosmology']:
    script = os.path.join(sp_base, 'sp_validation', sc)
    %run $script
"""
