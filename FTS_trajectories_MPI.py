###########################################################################
# This code performs parallelized FTS for different polymer densities,    #
# suitable for running on a cluster. The output files contain             #
# the chemical potential and osmotic pressure as a function  of CL time.  #
# The code is a part of the publication:                                  #
#                                                                         #
#    Y.-H. Lin, J. Wessén, T. Pal, S. Das and H.S. Chan (2022)            #
#    Numerical Techniques for Applications of Analytical Theories to      #
#    Sequence-Dependent Phase Separations of Intrinsically Disordered     #
#    Proteins.                                                            #
#    In: Phase-Separated Biomolecular Condensates, Methods and Protocols; #
#    Methods Mol. Biol;                                                   #
#    Zhou, H.-X., Spille, J.-H., Banerjee, P. R., Eds.;                   #
#    Springer-Nature, 2022; Vol. 2563, Chapter 3, pp 51−94,               #
#    DOI: 10.1007/978-1-0716-2663-4_3.                                    #
#    ( Pre-print available at arXiv:2201.01920 )                          #
#                                                                         #
# and follows the methods described therein.                              #
###########################################################################

from FTS_polyampholytes import *
import CL_seq_list as sl
import multiprocessing as mp
import sys
from joblib import Parallel, delayed


def exe(PS, run_label):

    np.random.seed()

    # CL time step
    dt = 0.01  # Complex-Langevin time step
    t_prod = int(5e4)  # Total number of time steps

    # Store parameter values
    params = PS.get_params()
    with open(run_label + "_params.txt", "w") as f:
        print(params, file=f)

    # Initialize fields as random fluctuations around mean field solution
    init_size = 0.1
    w = init_size * (
        np.random.randn(PS.Nx, PS.Nx, PS.Nx)
        + 1j * np.random.randn(PS.Nx, PS.Nx, PS.Nx)
    )
    psi = init_size * (
        np.random.randn(PS.Nx, PS.Nx, PS.Nx)
        + 1j * np.random.randn(PS.Nx, PS.Nx, PS.Nx)
    )

    w -= np.mean(w) + 1j * PS.rhop0 * PS.v
    psi -= np.mean(psi)
    PS.set_fields(w, psi)

    # Compute the M^(-1) matrix used for semi-implicit Complex Langevin evolution
    Minv = get_M_inv(PS, dt)

    with open(run_label + "_traj.txt", "w") as ev_file:
        # Generate Complex Langevin evolution trajectory for the chemical potential and osmotic pressure
        for t in range(t_prod):
            # Sample every 50th step
            if t % 50 == 0:
                mu = PS.get_chem_pot()
                Pi = PS.get_pressure()

                print(t, mu.real, mu.imag, Pi.real, Pi.imag)

                ev_file.write(
                    "{:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e}".format(
                        t, t * dt, mu.real, mu.imag, Pi.real, Pi.imag
                    )
                )
                ev_file.write("\n")
                ev_file.flush()
            CL_step_SI(PS, Minv, dt, useSI=True)


if __name__ == "__main__":
    ncpus = 8
    # All bead bulk densities considered.
    all_rho = np.exp(np.linspace(np.log(1e-6), np.log(10), ncpus))

    # All values of the Bjerrum length (lB)
    all_lB = 1.0 / np.linspace(1.0, 6.0, 5)
    lB_index = int(sys.argv[1])  # Index of the lB value for this run

    # Bjerrum length (inverse of the reduced temperature)
    lB = all_lB[lB_index]
    v = 0.0068  # Excluded volume parameter
    a = 1.0 / np.sqrt(6.0)  # Smearing length

    sig, N, the_seq = sl.get_the_charge("sv20")

    run_label_base = "data/sv20_lB_" + str(lB_index) + "_"

    procs = Parallel(n_jobs=ncpus)(
        delayed(exe)(
            PolySol(sig, rho, lB, v, a, Nx=24), run_label_base + str(run)
        )
        for run, rho in enumerate(all_rho)
    )

    for proc in procs:
        proc.join()
