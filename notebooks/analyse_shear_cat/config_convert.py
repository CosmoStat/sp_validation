import os

input_base = ['tutorial_UNIONS_SP_v1.0']

inp_str = '.ipynb '.join(input_base)


cmd = f'jupyter nbconvert --to script {inp_str} --stdout > tutorial_UNIONS_SP_v1.0.py'

os.system(cmd)

# Run validation from command line via
# ipython <script>.py
