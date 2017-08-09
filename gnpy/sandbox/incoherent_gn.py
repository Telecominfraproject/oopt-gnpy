# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:16:49 2016

@author: briantaylor
"""
from numpy import exp, pi
import numpy as np
from scipy.integrate import dblquad


def ign_rho(f, span, f1, f2):
    """
    This form or \\rho assumes lumped EDFA-like amplifcation.  This function is
    also known as the link function.

    Inputs:
    f = frequency array
    f1, f2  frequency bounds used to create a grid.
    span = the span object as defined by the Span class.
    ign_rho expects several parameters from Span in order to calculate the
    \\rho function.

    This form currently sets \\beta_3 in the denominator to zero.  This
    equation is taken from EQ[6], page 103 of:

    The GN-Model of Fiber Non-Linear Propagation and its Applications
    P. Poggiolini;G. Bosco;A. Carena;V. Curri;Y. Jiang;F. Forghieri (2014)

    Version used for this code came from:
    http://porto.polito.it/2542088/

    TODO:  Update the docu string with the IGN rho in Latex form
    TODO:  Fix num length

    """
    num = 1 - exp(-2 * span.alpha * span.length) * \
        exp((1j * 4 * pi**2) * (f1 - f) * (f2 - f) * span.beta2 * span.length)
    den = 2 * span.alpha - (1j * 4 * pi**2) * (f1 - f) * (f2 - f) * span.beta2
    rho = np.abs(num / den) * span.leff**-2
    return rho


def ign_function(f, span, f1, f2):
    """
    This creates the integrand for the incoherenet gaussian noise reference
    function (IGNRF).  It assumes \\rho for lumped EDFA-like amplifcation.

    Inputs:
    f = frequency array
    f1, f2  frequency bounds used to create a grid.
    span = the span object as defined by the Span class.

    This
    equation is taken from EQ[11], page 104 of:
    The GN-Model of Fiber Non-Linear Propagation and its Applications
    P. Poggiolini;G. Bosco;A. Carena;V. Curri;Y. Jiang;F. Forghieri (2014)

    Version used for this code came from:
    http://porto.polito.it/2542088/

    TODO:  Update the docu string with the IGN rho in Latex form
    """
    mult_coefs = 16 / 27 * (span.gamma ** 2) * span.nchan
    ign = mult_coefs * span.psd(f1) * span.psd(2)*span.psd(f1 + f2 - f) * \
        ign_rho(f, span, f1, f2)
    return ign


def integrate_ign(span, f1, f2, f, options=None):
    """
    integrate_ign integrates the ign function defined in ign_function.
    It uses scipy.integrate.dblquad to perform the double integral.

    The GN model is integrated over 3 distinct regions and the result is then
    summed.
    """

    """
    func : callable
    A Python function or method of at least two variables: y must be the first
    argument and x the second argument.
    a, b : float
    The limits of integration in x: a < b
    gfun : callable
    The lower boundary curve in y which is a function taking a single floating
    point argument (x) and returning a floating point result: a lambda function
    can be useful here.
    hfun : callable
    The upper boundary curve in y (same requirements as gfun).
    args : sequence, optional
    Extra arguments to pass to func.
    epsabs : float, optional
    Absolute tolerance passed directly to the inner 1-D quadrature integration.
    Default is 1.49e-8.
    epsrel : float, optional
    Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.

    dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08)



    Definition : quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08,
                      epsrel=1.49e-08, limit=50, points=None, weight=None,
                      wvar=None, wopts=None, maxp1=50, limlst=50)


    r1 = integral2(@(f1,f2) incoherent_inner(f, link, f1, f2),...
        max_lower_bound, max_upper_bound, mid_lower_bound, mid_upper_bound, ...
        'RelTol', model.RelTol, 'AbsTol', model.AbsTol_incoherent);

    """

    max_lower_bound = np.min(span.psd)
    max_upper_bound = np.max(span.psd)
    mid_lower_bound = f - span.model.bound
    mid_upper_bound = f + span.model.bound

    return [max_lower_bound, max_upper_bound, mid_lower_bound, mid_upper_bound]


def integrate_hyperbolic(span, f1, f2, f, options=None):
    return None
