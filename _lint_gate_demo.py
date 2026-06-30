"""Throwaway file to demo the develop lint gate. Delete me."""

import os  # F401: unused import


def demo():
    return undefined_thing + 1  # F821: undefined name
