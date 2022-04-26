import os

input_base = ['main_set_up']

inp_str = '.ipynb '.join(input_base)


cmd = f'jupyter nbconvert --to script {inp_str} --stdout > main_set_up.py'

os.system(cmd)

# Run from command line via
# ipython main_set_up.py

"""
sp_base = f"{os.environ['HOME']}/sp_validation"

for sc in ['cosmology']:
    script = os.path.join(sp_base, 'sp_validation', sc)
    %run $script
"""
