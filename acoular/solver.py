

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
    Bool
)
from traits.trait_errors import TraitError


# acoular imports
from .configuration import config
from .fastFuncs import damasSolverGaussSeidel
from .internal import digest

sklearn_ndict = {}
if parse(sklearn.__version__) < parse('1.4'):
    sklearn_ndict['normalize'] = False  # pragma: no cover

BEAMFORMER_BASE_DIGEST_DEPENDENCIES = ['freq_data.digest', 'r_diag', 'r_diag_norm', 'precision', 'steer.digest']

if config.have_pylops:
    import pylops


class SolverBase(HasStrictTraits):

    output = Any  # for storing the result of the solver, e.g., for debugging or analysis    
    options = Dict  # for storing options for the solver, e.g., for debugging or analysis
    digest = Property(depends_on=['options'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def solve(self, A, y, x, index):
        return NotImplementedError("The solve method must be implemented by subclasses of SolverBase.")


class GaussSeidelSolver(SolverBase):

    options = Dict({
        'niter': 100, 'damp': 1.0, 'callback': None, 'stol': 1e-10, 'rtol': 1e-10})

    def solve(self, A, y, x, index=None):
        niter = self.options.get('niter', 100)
        damp = self.options.get('damp', 1.0)
        stol = self.options.get('stol', 1e-10)
        rtol = self.options.get('rtol', 1e-10)
        ngrid = A.shape[1]
        if self.options.get('callback') is not None:
            callback = self.options['callback']
            have_callback = True
        else:
            have_callback = False
        cost = []
        s_cost = []
        r_cost = []
        y_norm = np.linalg.norm(y)
        for i in range(niter):
            x_old = x.copy()
            x = damasSolverGaussSeidel(A, y, ngrid, damp, x)
            if have_callback:
                callback(x)

            res = np.linalg.norm(A@x - y)
            sres = np.linalg.norm(x - x_old) / np.linalg.norm(x)
            rres = res / y_norm
            cost.append(res)
            s_cost.append(sres)
            r_cost.append(rres)
            if sres < stol:
                print(f"Converged at iteration {i+1} with sres={sres:.2e} < stol={stol:.2e}")
                break
            if rres < rtol:
                print(f"Converged at iteration {i+1} with rres={rres:.2e} < rtol={rtol:.2e}")
                break
        self.output = {
            index: {
                'ntotal': i + 1,
                'cost': cost,
                's_cost': s_cost,
                'r_cost': r_cost
            }
        }
        return x


class ISTACV(SolverBase):

        """
        Iterative Shrinkage-Thresholding Algorithm with Cross-Validation for regularization parameter selection.
    
        This is a custom solver that can be used with the BeamformerCMF class for covariance matrix fitting.
        """

        method = Enum('ISTA', 'FISTA')

        #: Number of grid points for cross-validation
        num_grid = Int(20) 

        #: Ratio of regularization parameter to the maximum value
        reg_ratio = Float(1e-3)  

        #: Number of cross-validation folds
        cv = Int(5)  

        #: Random seed for reproducibility
        seed = Int(42)

        cv_options = Union(None, Dict)  # for storing options for cross-validation, e.g., for debugging or analysis

        warm_start = Bool(True)  # whether to use the solution from the previous regularization parameter as the initial guess for the next one

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
                    cb1 = CallbackISTA(stol=options.get('stol', 1e-10), rtol=options.get('tol', 1e-10))
                    cb = [cb1]
                    if options.get('callback') is not None:
                        cb.append(options.pop('callback'))
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
            cb1 = CallbackISTA(stol=self.options.get('stol', 1e-10), rtol=self.options.get('tol', 1e-10))
            cb = [cb1]
            if self.options.get('callback') is not None:
                cb.append(self.options.pop('callback'))
            model_instance = model(pylops.MatrixMult(A), callbacks=cb)
            xhat, ntotal, cost = model_instance.solve(y, eps=eps, **self.options)
            self.output = {index:{
                'ntotal': ntotal,
                'cost': cost,
                'eps': eps,
                'r_cost': cb1.r_cost,
                's_cost': cb1.s_cost,
                'res_cost': cb1.cost

            }}
            return xhat.squeeze()
        

class CallbackISTA(pylops.optimization.callback.Callbacks):
    """
    see also:https://pylops.readthedocs.io/en/v2.5.0/tutorials/classsolvers.html#sphx-glr-tutorials-classsolvers-py
    """
    def __init__(self, stol=1e-10, rtol=1e-10):
        self.stol = stol
        self.rtol = rtol
        self.cost = []
        self.s_cost = []
        self.r_cost = []
        self.stop = False

    def on_run_begin(self, solver, x):
        """Callback before entire solver run

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x : :obj:`numpy.ndarray`
            Current model vector

        """
        self.y_norm = np.linalg.norm(solver.y)
        self.x0 = x.copy()

    def on_step_end(self, solver, x):
        if solver.iiter <= 1:
            self.xold = self.x0
            return  # skip first iteration since it is just the initial guess
        res = np.linalg.norm(solver.Op @ x - solver.y)
        sres = np.linalg.norm(x - self.xold) / np.linalg.norm(x)
        rres = res / self.y_norm
        if self.xold is not None:
            self.cost.append(res)
            self.s_cost.append(sres)
            self.r_cost.append(rres)
        self.xold = x
        if sres < self.stol:
            print(f"Converged at iteration {solver.iiter} with sres={sres:.2e} < stol={self.stol:.2e}")
            self.stop = True
        if rres < self.rtol:
            print(f"Converged at iteration {solver.iiter} with rres={rres:.2e} < rtol={self.rtol:.2e}")
            self.stop = True

