"""
This file is part of FNFTpy.
FNFTpy provides wrapper functions to interact with FNFT,
a library for the numerical computation of nonlinear Fourier transforms.

For FNFTpy to work, a copy of FNFT has to be installed.
For general information, source files and installation of FNFT,
visit FNFT's github page: https://github.com/FastNFT

For information about setup and usage of FNFTpy see README.md or documentation.

FNFTpy is free software; you can redistribute it and/or
modify it under the terms of the version 2 of the GNU General
Public License as published by the Free Software Foundation.

FNFTpy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Contributors:

Christoph Mahnke, 2018

"""
import numpy as np

import thesis.src.visualizer
from FNFTpy.FNFTpy import nsev


def nsev_example():
    """Mimics the C example for calling fnft_nsev."""
    print("\n\nnsev example")

    # set values
    D = 256
    tvec = np.linspace(-1, 1, D)
    q = np.zeros(len(tvec), dtype=np.complex128)
    q[:] = 2.0 + 0.0j
    M = 8
    Xi1 = -2
    Xi2 = 2
    Xivec = np.linspace(Xi1, Xi2, M)

    # call function
    res = nsev(q, tvec, M=M, Xi1=Xi1, Xi2=Xi2)

    # print results
    print("\n----- options used ----")
    print(res['options'])
    print("\n------ results --------")

    print("FNFT return value: %d (should be 0)" % res['return_value'])
    print("continuous spectrum")
    for i in range(len(res['cont_ref'])):
        print("%d :  Xi = %.4f   %.6f  %.6fj" % (
            i, Xivec[i], np.real(res['cont_ref'][i]), np.imag(res['cont_ref'][i])))
    print("discrete spectrum")
    for i in range(len(res['bound_states'])):
        print("%d : %.6f  %.6fj with norming const %.6f  %.6fj" % (i, np.real(res['bound_states'][i]),
                                                                   np.imag(res['bound_states'][i]),
                                                                   np.real(res['disc_norm'][i]),
                                                                   np.imag(res['disc_norm'][i])))

    thesis.src.visualizer.my_plot(Xivec, np.abs(res['cont_ref']), title='continues spectrum |X[xi]|',
                                  output_name='nsev_cont_absXi.jpg', xlabel='xi')
    thesis.src.visualizer.my_plot(Xivec, np.abs(res['disc_norm']), title='discrete spectrum |X[xi]|',
                                  output_name='nsev_disc_absXi.jpg', xlabel='xi')
    return res


nsev_example()
