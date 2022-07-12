import os

input_base = ['xip_xim']

inp_str = '.ipynb '.join(input_base)


cmd = f'jupyter nbconvert --to script {inp_str} --stdout > xip_xim.py'

os.system(cmd)

# Run validation from command line via
# ipython <script>.py
