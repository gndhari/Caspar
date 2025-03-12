#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# sun_reconstruction.py: Functions to perform reconstruction of SU(n)
#                        transformations using a list of modes and parameters.
#
# © 2017 Olivia Di Matteo (odimatte@uwaterloo.ca)
#
# This file is part of the project Caspar.
# Licensed under BSD-3-Clause
#

import numpy as np
import autograd.numpy as anp

def embed_su2(n, i, j, params):
    """ Embed the SU(2) transformation given by params into modes i and j
        of an SU(n) matrix
            SU_ij(3) = [ e^(i(a+g)/2) cos(b/2)   -e^(i(a-g)/2) sin(b/2)
                        e^(-i(a-g)/2) sin(b/2)   e^(-i(a-g)/2) cos(b/2) ]

        Returns the full n-dimensional matrix.
    """
    a, b, g = params[0], params[1], params[2]

    # Create SU(2) element and scaled by loss if desired.
    Rij = anp.array([[anp.exp(1j*(a+g)/2)*anp.cos(b/2), -anp.exp(1j*(a-g)/2)*anp.sin(b/2)],
                    [anp.exp(-1j*(a-g)/2)*anp.sin(b/2), anp.exp(-1j*(a+g)/2)*anp.cos(b/2)]])

    # Stuff it into modes i and j of SU(n)
    # full_Rij = anp.asmatrix(anp.identity(n)) + 0j
    # full_Rij[i:j+1, i:j+1] = Rij

    full_Rij = anp.identity(n, dtype=complex)  # Use anp.array() instead of asmatrix()

    # Manually replace the submatrix without in-place modification
    full_Rij = anp.concatenate([
        full_Rij[:i, :],  # Keep the top part
        anp.concatenate([
            full_Rij[i:j+1, :i],  # Keep the left part of the submatrix
            Rij,                  # Replace the submatrix
            full_Rij[i:j+1, j+1:]  # Keep the right part of the submatrix
        ], axis=1),
        full_Rij[j+1:, :]  # Keep the bottom part
    ], axis=0)

    return full_Rij


def sun_reconstruction(n, parameters):
    """ Reconstruct an SU(n) matrix using a list of transformations given as
        tuples ("i,j", [a, b, g]). In theory any sequence can be given, but
        it would make most sense to use something of the form of the sequence
        of n(n-1)/2 transformations given by sun_factorization algorithm.

        Note: the matrix is put together "backwards" as compared to the
        circuit diagram, i.e. we multiply on the right always by the next
        matrix to get U, rather than on the left. In the end it really
        doesn't matter since the decomposition itself is symmetric, as long
        as everything is consistent.
    """

    # Hold the result
    U = anp.array(anp.identity(n)) + 0j

    for param in parameters:
        # Get the indices of the modes
        modes = param[0].split(",")
        md1, md2 = int(modes[0]) - 1, int(modes[1]) - 1

        if md1 not in range(n) or md2 not in range(n):
            print("Mode combination " + str(md1 + 1) + "," + str(md2 + 1) + \
                    " is invalid for a system of dimension " + str(n) + ".")
            return

        if md2 != md1 + 1:
            print("Mode combination " + str(md1 + 1) + "," + str(md2 + 1) + " is invalid. ")
            print("Currently only transformations on adjacent modes are implemented.")
            return

        # Compute the next transformation and multiply
        next_trans = embed_su2(n, md1, md2, anp.array(param[1])) # removed anp.array() for RHS
        U = U @ next_trans

    return U


def sun_reconstruction_mod(n, parameters, breakpoints):
    # Hold the result
    U = anp.array(anp.identity(n)) + 0j

    for param, brk in zip(parameters, breakpoints):
        # Get the indices of the modes
        md1, md2 = param[0] - 1, param[1] - 1

        if md1 not in range(n) or md2 not in range(n):
            print("Mode combination " + str(md1 + 1) + "," + str(md2 + 1) + \
                    " is invalid for a system of dimension " + str(n) + ".")
            return

        if md2 != md1 + 1:
            print("Mode combination " + str(md1 + 1) + "," + str(md2 + 1) + " is invalid. ")
            print("Currently only transformations on adjacent modes are implemented.")
            return

        # Compute the next transformation and multiply
        next_trans = anp.array(embed_su2(n, md1, md2, anp.array(brk))) # removed anp.array() for RHS
        U = U @ next_trans

    return U
