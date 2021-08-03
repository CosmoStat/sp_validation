import os

input_base = ['main_set_up', 'metacal_global', 'psf_leakage']

inp_str = '.ipynb '.join(input_base)


cmd = f'jupyter nbconvert --to script {inp_str} --stdout > validation.py'

os.system(cmd)

# Run validation from command line via
# ipython validation.py
