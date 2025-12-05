# %%
import numpy as np
import camb
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.cosmology import Planck18
from astropy.io import fits
from scipy.interpolate import interp1d

plt.style.use(
    '/home/guerrini/matplotlib_config/paper.mplstyle'
)
plt.rcParams['text.usetex'] = True
sns.set_palette("husl")

# %%
path_redshift_distr = "/n17data/sguerrini/UNIONS/WL/nz/v1.4.6/nz_SP_v1.4.6_A.txt"

z, dndz = np.loadtxt(path_redshift_distr, unpack=True)
n_z = interp1d(z, dndz, bounds_error=False, fill_value=0)

# %%
plt.figure()

plt.plot(z, dndz)

plt.xlabel('Redshift z')
plt.ylabel(r'$dN/dz$')

plt.show()

# %%
#Define a Planck cosmology object with CAMB

planck = Planck18

h = planck.H0.value/100
Om = planck.Om0
Ob = planck.Ob0
Oc = Om - Ob
ns = 0.965
As = 2.1e-9
m_nu = 0.06
w = -1

pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
Onu = pars.omeganu
Oc = Om - Ob - Onu
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)


# %%
camb_results = camb.get_results(pars)


# %%
# Non-linear matter power interpolator
PK = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=True, k_hunit=True, kmax=20.0, zmax=5.0)

# %%
# ---- Comoving distance <-> redshift mapping ----
chi_of_z = np.vectorize(lambda z: planck.comoving_distance(z).value) # [Mpc]
z_of_chi = interp1d(chi_of_z(np.linspace(0,5,2000)), np.linspace(0,5,2000), fill_value="extrapolate")

# %%
# ---- Example lensing efficiency function q^i(chi) ----
def q_i(chi, n_i):
    """
    Example lensing efficiency function for source distribution n_i(chi).
    n_i(chi) should be normalized.
    """
    a = 1.0 / (1.0 + z_of_chi(chi))
    prefactor = 1.5 * planck.Om0 * (planck.H0.value/299792.458)**2 * chi / a
    chi_vals = np.linspace(chi, chi_of_z(5.0), 2000)
    integrand = n_i(chi_vals) * (chi_vals - chi) / chi_vals
    return prefactor * np.trapz(integrand, chi_vals)


# ---- Example galaxy redshift distribution n_i(chi) ----
def n_i_example(chi):
    z = z_of_chi(chi)
    return n_z(z) # arbitrary form (Smail-like)


# ---- Integrand function ----
def Cl_integrand(k, ell, q_i_func, q_j_func):
    chi = (ell + 0.5) / k
    z = float(z_of_chi(chi))
    q_i_val = q_i_func(chi, n_i_example)
    q_j_val = q_j_func(chi, n_i_example)
    Pk = PK.P(z, k) # nonlinear P(k,z)
    return (k / (ell + 0.5)) * q_i_val * q_j_val * Pk

# %%
# ---- Example usage ----
ell_list = [30, 100, 300, 1000, 3000]
k_vals = np.logspace(-3, 2, 5000)
I_vals_list = []

for ell in ell_list:
    Cl_function = np.vectorize(lambda k: Cl_integrand(k, ell, q_i, q_i))
    I_vals_list.append(Cl_function(k_vals))


# %%
plt.figure()

for ell, I_vals in zip(ell_list, I_vals_list):
    plt.plot(k_vals, I_vals/np.max(I_vals), label=f'$\ell={ell}$')
    print(np.trapz(I_vals, k_vals))

plt.xscale('log')
plt.xlabel(r'$k$ [h/Mpc]', fontsize=12)
plt.ylabel(r'$\frac{\mathrm{d}C_\ell}{\mathrm{d}\ln k}$ (normalized)', fontsize=12)
plt.yticks([0.])
plt.legend(fontsize=12)
plt.savefig('./plots/Cl_integrand_vs_k.png', dpi=300)
# Save pdf
plt.savefig('./plots/Cl_integrand_vs_k.pdf')
plt.show()


# %%
# Plot lensing efficiency
z_lensing = np.linspace(0, 3, 2000)
chi = chi_of_z(z_lensing)
q_vals = [q_i(chi, n_i_example) for chi in chi]

plt.figure()

plt.plot(z_lensing, q_vals/np.trapz(q_vals[1:], z_lensing[1:]), label=r'$q^i(z)$')
plt.plot(z, dndz/np.trapz(dndz, z), linestyle="dashed", alpha=0.5, color='C0')

plt.xlabel('Redshift z')
plt.ylabel(r'$q^i(z)$')

plt.show()

# %%
def get_kalpha(I_vals, k_vals, alpha):
    """
    Computes the k-scale below which alpha percent of the total Cl signal is contained.
    """
    total_signal = np.trapz(I_vals, k_vals)
    target_signal = alpha * total_signal
    cumulative_signal = np.cumsum(I_vals) * np.diff(k_vals, prepend=0)
    k_alpha = k_vals[np.searchsorted(cumulative_signal, target_signal)]
    return k_alpha
# %%
alpha = 0.95
k_alpha_list = [get_kalpha(I_vals, k_vals, alpha) for I_vals in I_vals_list]
# %%
print(k_alpha_list)

# %%
def get_lmax(k_max, alpha, k_vals, l_low=400, l_high=2048):
    """
    Computes the maximum multipole l_max corresponding to a given k_max and alpha percent of the total Cl signal using dichotomy.
    """
    def objective(ell):
        k_alpha = get_kalpha(np.vectorize(lambda k: Cl_integrand(k, ell, q_i, q_i))(k_vals), k_vals, alpha)
        print(k_alpha)
        return k_alpha - k_max

    n_iter = 1
    while l_high - l_low > 1:
        print(f"Iteration {n_iter}: l_low={l_low}, l_high={l_high}")
        l_mid = (l_low + l_high) // 2
        if objective(l_mid) > 0:
            l_high = l_mid
        else:
            l_low = l_mid
        n_iter += 1

    return l_low if objective(l_low) <= 0 else l_high
    
# %%
get_lmax(3, 0.95, k_vals, l_high=4096)
# %%
get_lmax(1, 0.95, k_vals, l_high=4096)
# %%
get_lmax(5, 0.95, k_vals, l_high=4096)
# %%
