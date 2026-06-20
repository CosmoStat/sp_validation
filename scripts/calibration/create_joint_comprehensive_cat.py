#!/usr/bin/env python3

import sys

from sp_validation.catalog_builders import run_joint_comprehensive_cat


def main(argv=None):

    if argv is None:
        argv = sys.argv[0:]
    run_joint_comprehensive_cat(*argv)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
