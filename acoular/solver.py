

import numpy as np

# check for sklearn version to account for incompatible behavior
import sklearn
from packaging.version import parse
from traits.api import (
    Any,
    Dict,
    Enum,
    Float,
    HasStrictTraits,
    Int,
    Union,
    Property,
    cached_property,
    Bool,
    Instance,
    List
)
from traits.trait_errors import TraitError
from scipy.linalg import solve


# acoular imports
from .configuration import config
from .fastFuncs import damasSolverGaussSeidel
from .internal import digest
from warnings import warn

sklearn_ndict = {}
if parse(sklearn.__version__) < parse('1.4'):
    sklearn_ndict['normalize'] = False  # pragma: no cover

BEAMFORMER_BASE_DIGEST_DEPENDENCIES = ['freq_data.digest', 'r_diag', 'r_diag_norm', 'precision', 'steer.digest']

if config.have_pylops:
    import pylops


class SolverBase(HasStrictTraits):

    output = Dict  # for storing the result of the solver, e.g., for debugging or analysis    
    options = Dict  # for storing options for the solver, e.g., for debugging or analysis
    callback = Union(None, Any)
    digest = Property(depends_on=['options','callback'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def solve(self, A, y, x, index):
        return NotImplementedError("The solve method must be implemented by subclasses of SolverBase.")


class GaussSeidelSolver(SolverBase):

    options = Dict({
        'niter': 100, 'damp': 1.0, 'stol1': 1e-10, 'stol': 1e-10, 'rtol': 1e-10})

    def solve(self, A, y, x, index=None):
        niter = self.options.get('niter', 100)
        damp = self.options.get('damp', 1.0)
        stol = self.options.get('stol', 1e-10)
        stol1 = self.options.get('stol1', 1e-10)
        rtol = self.options.get('rtol', 1e-10)
        ngrid = A.shape[1]
        cost = []
        s_cost = []
        s1_cost = []
        r_cost = []
        y_norm = np.linalg.norm(y)
        for i in range(niter):
            x_old = x.copy()
            x = damasSolverGaussSeidel(A, y, ngrid, damp, x)
            if self.callback is not None:
                self.callback(x)

            res = np.linalg.norm(A@x - y)
            sres = np.linalg.norm(x - x_old) / np.linalg.norm(x)
            sres1 = np.linalg.norm(x - x_old, ord=1) / np.linalg.norm(x_old, ord=1)
            rres = res / y_norm
            cost.append(res)
            s_cost.append(sres)
            s1_cost.append(sres1)
            r_cost.append(rres)
            if sres < stol:
                print(f"Converged at iteration {i+1} with sres={sres:.2e} < stol={stol:.2e}")
                break
            if rres < rtol:
                print(f"Converged at iteration {i+1} with rres={rres:.2e} < rtol={rtol:.2e}")
                break
            if sres1 < stol1:
                print(f"Converged at iteration {i+1} with sres1={sres1:.2e} < stol1={stol1:.2e}")
                break
        self.output = {
            index: {
                'ntotal': i + 1,
                'cost': cost,
                's_cost': s_cost,
                's1_cost': s1_cost,
                'r_cost': r_cost
            }
        }
        return x

class SBLSolver(SolverBase):

    method = Enum('SBL', 'SBL1', 'M-SBL')

    options = Dict({
        'niter': 100, 'stol': 1e-17, 'stol1': 1e-10, 'grange': 1e-3, 'nsources': None, 'sig_sq': None
        })

    def estimate_noise_power(self, gamma, csm, a):
        # 
        nc = a.shape[0]
        if self.options.get('nsources') is None:
            ga = np.argwhere(gamma>max(gamma)*self.options.get('grange', 1e-4)).ravel()
            ga_sum = ga.sum() # active set
            if ga_sum > nc - 1:
                # sort gamma and take the num_channels-1 largest values
                sorted_indices = np.argsort(gamma)[::-1]
                ga = np.zeros_like(gamma, dtype=bool)
                ga[sorted_indices[:nc - 1]] = True
        else:
            ga = self.find_peaks(gamma)
        a_sub = a[:,ga]
        P = a_sub @ np.linalg.solve(a_sub.conj().T @ a_sub, a_sub.conj().T)
        return np.real(np.trace((np.eye(nc) - P) @ csm / (nc - a_sub.shape[1])))

    def find_peaks(self, gamma):
        ns = self.options.get('nsources')
        locs = np.zeros((ns),dtype = int)
        ng = len(gamma)
        gamma  = gamma.reshape(ng)
        gamma_new = np.zeros((ng+2)) # zero padding on the boundary
        gamma_new[1:ng+1] = gamma;
        Ilocs  = np.flip(gamma.argsort(axis = 0))
        npeaks = 0
        local_patch=np.zeros((ns))
        for ii in range(ng):
            local_patch = [gamma_new[(Ilocs[ii])], 0, gamma_new[(Ilocs[ii]+2)]]
            # zero the center
            if sum(gamma[Ilocs[ii]] > local_patch) == 3:
                locs[npeaks] = Ilocs[ii]
                npeaks = npeaks + 1
                # if found sufficient peaks, break
                if npeaks == ns:
                    break
        return locs
    
    def solve(self, A, y, csm, gamma, index):
        # get solver options
        nc = A.shape[0]
        if self.options.get('stol') is not None:
            stol = self.options.pop('stol')
        else:
            stol = 1e-10
        if self.options.get('stol1') is not None:
            stol1 = self.options.pop('stol1')
        else:            
            stol1 = 1e-10
        niter = self.options.get('niter', 100)
        s_cost = []
        s1_cost = []

        if gamma is None:
            gamma = np.einsum('mg,mn,ng->g', A.conj(), csm, A).real
            gamma[gamma < 0] = 0
        I = np.eye(csm.shape[0])
        L = y.shape[0]
        
        # SBL uses sample cross-spectral matrix for the denominator (can be calculated once)
        if self.method == 'SBL':
            csm_inv_a_denum = solve(csm, A, assume_a='her')
            gamma_denum_full = np.abs(np.sum(np.conj(A) * csm_inv_a_denum, axis=0).real)
        
        
        if self.options.get('sig_sq') is not None:
            sig_sq_est = self.options.get('sig_sq')
            estimate_noise_variance = False
        else:
            max_sig_sq = np.real(np.trace(csm))/nc 
            sig_sq_est = max_sig_sq
            estimate_noise_variance = True

        for i in range(niter):
            gamma_prev = gamma.copy()
            ga = np.argwhere(gamma>max(gamma)*self.options.get('grange', 1e-4)).ravel()
            gamma_sub = gamma[ga]
            a_sub = A[:,ga]
            csm_mod = (a_sub * gamma_sub[None, :]) @ a_sub.conj().T + sig_sq_est * I
            csm_inv_a = solve(csm_mod, a_sub, assume_a='her')
            gamma_num = np.linalg.norm(y.conj() @ csm_inv_a, axis=0)**2 / L
            if self.method == 'SBL':
                gamma_denum = gamma_denum_full[ga]
            elif self.method == 'SBL1':
                gamma_denum = np.abs(np.sum(np.conj(a_sub) * csm_inv_a, axis=0).real)
            
            # calc gamma
            if self.method == 'M-SBL':
#                sigma_x = np.linalg.inv(1/sig_sq_est*a_sub.T.conj()@a_sub  + np.diag(1/gamma_sub))
                A_matrix = 1/sig_sq_est * a_sub.T.conj() @ a_sub + np.diag(1/gamma_sub)
                sigma_x_diag = np.linalg.solve(A_matrix, np.eye(A_matrix.shape[0])).diagonal()
                gamma[ga] = (gamma[ga]**2) * gamma_num + sigma_x_diag
            else:
                gamma[ga] *= np.sqrt(gamma_num / gamma_denum)  
            if np.any(gamma < 0):
                warn(f'Negative gamma values at iteration {i+1}, setting to zero.')
                gamma[gamma < 0] = 0

            if self.callback is not None:
                self.callback(gamma)

            # new noise estimate
            if estimate_noise_variance:
                sig_sq_est = np.minimum(self.estimate_noise_power(gamma, csm, A), max_sig_sq)
                sig_sq_est = max(sig_sq_est, 1e-10*max_sig_sq) 

            # checks convergence and displays status reports
            sres = np.linalg.norm(gamma - gamma_prev) / np.linalg.norm(gamma)
            sres1 = np.linalg.norm(gamma - gamma_prev, ord=1) / np.linalg.norm(gamma_prev, ord=1)
            if sres < stol:
                print(f"Converged at iteration {i+1} with sres={sres:.2e} < stol={stol:.2e}")
                break
            if sres1 < stol1:
                print(f"Converged at iteration {i+1} with sres1={sres1:.2e} < stol1={stol1:.2e}")
                break
            s_cost.append(sres)
            s1_cost.append(sres1)

        self.output = {
            index: {
                'ntotal': i + 1,
                's_cost': s_cost,
                's1_cost': s1_cost,
            }
        }
        return gamma    

class ISTACV(SolverBase):

        """
        Iterative Shrinkage-Thresholding Algorithm with Cross-Validation for regularization parameter selection.
    
        This is a custom solver that can be used with the BeamformerCMF class for covariance matrix fitting.
        """

        method = Enum('ISTA', 'FISTA')

        callback = Union(
            None, 
            Instance(pylops.optimization.callback.Callbacks),
            List(Instance(pylops.optimization.callback.Callbacks)))  
            # callback function for iterations, e.g., for debugging or analysis

        #: Number of grid points for cross-validation
        num_grid = Int(20) 

        #: Ratio of regularization parameter to the maximum value
        reg_ratio = Float(1e-3)  

        #: Random seed for reproducibility
        seed = Int(42)

        #: Number of cross-validation folds
        cv = Int(5)  

        cv_options = Union(None, Dict)  # for storing options for cross-validation, e.g., for debugging or analysis

        cv_callback = Union(None, Instance(pylops.optimization.callback.Callbacks))  # callback function for cross-validation iterations, e.g., for debugging or analysis

        warm_start = Bool(True)  # whether to use the solution from the previous regularization parameter as the initial guess for the next one

        positive = Bool(True)  # whether to enforce non-negativity constraint on the solution

        def grid_search(self, A, y, x0, reg, model):
            options = self.cv_options if self.cv_options is not None else self.options
            y = np.asarray(y).ravel()
            M = y.size
            kf = sklearn.model_selection.KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

            mean_val_losses = []
            reg = list(reg)
            x0 = x0.copy()  # Initial guess for ISTA

            for eps in reg:
                fold_losses = []
                for train_idx, val_idx in kf.split(np.arange(M)):
                    # Restrict measurements (rows) for train and val :contentReference[oaicite:3]{index=3}
                    Rtr = pylops.Restriction(dims=(M,), iava=train_idx)
                    Rva = pylops.Restriction(dims=(M,), iava=val_idx)
                    Optr = Rtr * A
                    ytr  = Rtr * y
                    Opva = Rva * A
                    yva  = Rva * y
                    # Solve on training subset
                    cb = []
                    if self.positive:
                        cb.append(PositivityCallback())
                    if self.cv_callback is not None:
                        if isinstance(self.cv_callback, list):
                            cb.extend(self.cv_callback)
                        else:
                            cb.append(self.cv_callback)
                    model_instance = model(pylops.MatrixMult(Optr), callbacks=cb)
                    xhat, _, _ = model_instance.solve(ytr,eps=eps, x0=x0, **options)

                    # Validate: data-fit on held-out measurements
                    resid = (Opva @ xhat) - yva
                    val_mse = (np.vdot(resid, resid).real) / len(val_idx)  # MSE on validation measurements
                    fold_losses.append(val_mse)
                if self.warm_start:
                    x0 = xhat  # warm start for next iteration
                fold_losses = np.asarray(fold_losses)
                mean_val_losses.append(fold_losses.mean())

            return reg[int(np.argmin(np.asarray(mean_val_losses)))]

        def solve(self, A, y, x, index):
            if self.method == 'FISTA':
                model = pylops.optimization.sparsity.FISTA
            elif self.method == 'ISTA':
                model = pylops.optimization.sparsity.ISTA
            else:
                raise TraitError(f"Unsupported method: {self.method}")

            reg_max = np.max(np.abs(A.T @ y))
            reg_values = np.geomspace(reg_max, reg_max * self.reg_ratio, self.num_grid)
            eps = self.grid_search(A, y, x, reg_values, model)

            # run main loop
            cb = []
            if self.positive:
                cb.append(PositivityCallback())
            if self.callback is not None:
                if isinstance(self.callback, list):
                    cb.extend(self.callback)
                else:
                    cb.append(self.callback)
            cb = cb[::-1] # reverse order so that main callback is last and thus called first in pylops
            model_instance = model(pylops.MatrixMult(A), callbacks=cb)
            xhat, ntotal, cost = model_instance.solve(y, eps=eps, **self.options)
            
            # Build output dict with available cost data
            output_dict = {
                'ntotal': ntotal,
                'cost': cost,
                'eps': eps,
            }
            self.output = {index: output_dict}
            return xhat.squeeze()
        

class CallbackBase(pylops.optimization.callback.Callbacks):
    """
    Base class for callbacks that monitor convergence during optimization.
    
    Attributes
    ----------
    tol : float
        Tolerance for convergence criterion
    cost : list
        List to store the cost values at each iteration
    stop : bool
        Flag to indicate if the solver should stop
    """
    def __init__(self, tol=1e-10):
        self.tol = tol
        self.cost = []
        self.stop = False


class SolutionResidualCallback(CallbackBase):
    """
    Callback to monitor convergence based on solution residual (L2 norm).
    
    Monitors the relative change in the solution vector between iterations
    using the L2 norm: ||x - x_old|| / ||x||
    
    Attributes
    ----------
    tol : float
        Tolerance for solution residual convergence
    """
    def on_run_begin(self, solver, x):
        self.x0 = x.copy()
    
    def on_step_end(self, solver, x):
        if solver.iiter <= 1:
            self.xold = self.x0
            return
        sres = np.linalg.norm(x - self.xold) / np.linalg.norm(x)
        self.cost.append(sres)
        self.xold = x
        if sres < self.tol:
            print(f"Converged at iteration {solver.iiter} with sres={sres:.2e} < tol={self.tol:.2e}")
            self.stop = True


class SolutionResidualL1Callback(CallbackBase):
    """
    Callback to monitor convergence based on solution residual (L1 norm).
    
    Monitors the relative change in the solution vector between iterations
    using the L1 norm: ||x - x_old||_1 / ||x_old||_1
    
    Attributes
    ----------
    tol : float
        Tolerance for L1 solution residual convergence
    """
    def on_run_begin(self, solver, x):
        self.x0 = x.copy()
    
    def on_step_end(self, solver, x):
        if solver.iiter <= 1:
            self.xold = self.x0
            return
        sres1 = np.linalg.norm(x - self.xold, ord=1) / np.linalg.norm(self.xold, ord=1)
        self.cost.append(sres1)
        self.xold = x
        if sres1 < self.tol:
            print(f"Converged at iteration {solver.iiter} with sres1={sres1:.2e} < tol={self.tol:.2e}")
            self.stop = True


class RelativeResidualCallback(CallbackBase):
    """
    Callback to monitor convergence based on relative residual.
    
    Monitors the relative residual: ||A @ x - y|| / ||y||
    
    Attributes
    ----------
    tol : float
        Tolerance for relative residual convergence
    """
    def on_run_begin(self, solver, x):
        self.y_norm = np.linalg.norm(solver.y)
    
    def on_step_end(self, solver, x):
        if solver.iiter <= 1:
            return
        res = np.linalg.norm(solver.Op @ x - solver.y)
        rres = res / self.y_norm
        self.cost.append(rres)
        if rres < self.tol:
            print(f"Converged at iteration {solver.iiter} with rres={rres:.2e} < tol={self.tol:.2e}")
            self.stop = True


class AbsoluteResidualCallback(CallbackBase):
    """
    Callback to monitor absolute residual.
    
    Monitors the absolute residual: ||A @ x - y||
    
    Attributes
    ----------
    tol : float
        Tolerance for absolute residual
    """
    def on_step_end(self, solver, x):
        if solver.iiter <= 1:
            return
        res = np.linalg.norm(solver.Op @ x - solver.y)
        self.cost.append(res)
        if res < self.tol:
            print(f"Converged at iteration {solver.iiter} with res={res:.2e} < tol={self.tol:.2e}")
            self.stop = True


class PositivityCallback(pylops.optimization.callback.Callbacks):
    """
    Callback to enforce non-negativity constraint on the solution during optimization.
    """
    # def on_step_end(self, solver, x):
    #     x[...] = np.maximum(x, 0)  # Enforce non-negativity
    #     return x

    def on_step_begin(self, solver, x):
        x[...] = np.maximum(x, 0)  # Enforce non-negativity
        return x



class NNLSProjLandweber(SolverBase):

    positive = Bool(True)  # whether to enforce non-negativity constraint on the solution

    callback = Union(
        None, 
        Instance(pylops.optimization.callback.Callbacks), 
        List(pylops.optimization.callback.Callbacks))  # callback function for iterations, e.g., for debugging or analysis

    def solve(self, A, y, x, index):
        cb = []
        if self.positive:
            cb.append(PositivityCallback())

        if self.callback is not None:
            if isinstance(self.callback, list):
                cb.extend(self.callback)
            else:
                cb.append(self.callback)
        
        cb = cb[::-1] # pylops calls callbacks in reverse order
        model_instance = pylops.optimization.sparsity.ISTA(pylops.MatrixMult(A), callbacks=cb)
        xhat, ntotal, cost = model_instance.solve(y, eps=0.0, **self.options)
        
        # Build output dict with available cost data
        output_dict = {
            'ntotal': ntotal,
            'cost': cost,
        }
        self.output = {index: output_dict}
        return xhat.squeeze()