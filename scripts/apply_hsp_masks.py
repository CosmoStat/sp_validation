#!/usr/bin/env python3

import sys

from sp_validation import run_joint_cat as sp_joint


def main(argv=None):
    """Main

    Main program

    """
    if argv is None:
        argv = sys.argv[1:]
    sp_joint.run_apply_hsp_masks(*argv)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
