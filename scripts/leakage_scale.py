#!/usr/bin/env python3

"""LEAKAGE SCALE SCRIPT                                                         
                                                                                
This script computes the scale-dependent PSF leakage.
                                                                                
:Author: Martin Kilbinger <martin.kilbinger@cea.fr>                                 
                                                                                
"""  

import sys

from sp_validation.run import run_leakage_scale


def main(argv=None):
    """Main.

    Main program.

    """
    run_leakage_scale(*argv)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
