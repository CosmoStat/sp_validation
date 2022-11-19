import os

input_base = ['main_set_up', 'metacal_global', 'write_cat']

inp_str = '.ipynb '.join(input_base)


cmd = f'jupyter nbconvert --to script {inp_str} --stdout > validation_random.py'

os.system(cmd)

# Run validation from command line via
# ipython validation.py

"""
sp_base = f"{os.environ['HOME']}/sp_validation"

for sc in ['cosmology']:
    script = os.path.join(sp_base, 'sp_validation', sc)
    %run $script
"""
