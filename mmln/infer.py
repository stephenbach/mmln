import logging
import math


class Inference:

    def __init__(self):
        pass
    
    def set_potential(self, weight, coefficients, variables, constant, two_sided=False, squared=False):
        raise NotImplementedError('This class is abstract.')
    
    def infer(self):
        raise NotImplementedError('This class is abstract.')


class Variable:

    _counter = 1

    def __init__(self, id=None):
        self.value = 0
        self.id = id if id is not None else 'Internal ID #' + str(Variable._counter)
        Variable._counter += 1

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Variable<' + str(self.id) + '>'


class HLMRF(Inference):
    
    def __init__(self, eta=1.0, epsilon_abs=1e-8, epsilon_rel=1e-3, max_iter=25000):
        super(HLMRF, self).__init__()
        self.eta = eta
        self.epsilon_abs = epsilon_abs
        self.epsilon_rel = epsilon_rel
        self.max_iter = max_iter

        self.logger = logging.getLogger(__name__)
        
        self.pots = {}
        self.vars = set()
        self.needs_init = True

    def set_potential(self, weight, coefficients, variables, constant, two_sided=False, squared=False):
        if isinstance(coefficients, (int, float)):
            coefficients = (coefficients,)
        if isinstance(variables, Variable):
            variables = (variables,)

        if not squared:
            raise NotImplementedError('Only squared potentials are currently supported.')
        if len(coefficients) != len(variables):
            raise Exception('Must provide the same number of coefficients and variables.')
        if weight < 0:
            raise Exception('Only non-negative weights are allowed.')

        for var in variables:
            self.vars.add(var)

        key = (coefficients, variables, constant, two_sided, squared)
        if weight == 0:
            del self.pots[key]
            self.needs_init = True
        elif key not in self.pots:
            if len(variables) == 1:
                if two_sided:
                    self.pots[key] = _ADMMOneVarBowlPotential(self, weight, coefficients[0], variables[0], constant)
                else:
                    self.pots[key] = _ADMMOneVarHingePotential(self, weight, coefficients[0], variables[0], constant)
            elif len(variables) == 2:
                if two_sided:
                    self.pots[key] = _ADMMTwoVarBowlPotential(self, weight, coefficients[0], variables[0],
                                                              coefficients[1], variables[1], constant)
                else:
                    self.pots[key] = _ADMMTwoVarHingePotential(self, weight, coefficients[0], variables[0],
                                                               coefficients[1], variables[1], constant)
            else:
                raise Exception('Only potentials on one or two variables are supported.')
        else:
            self.pots[key].weight = weight

    def infer(self):
        self.logger.info('Starting optimization with ' + str(len(self.vars)) + ' variables and ' +
                         str(len(self.pots)) + ' potentials.')
        if self.needs_init:
            self._init()

        # Sets up
        primal_res = float('inf')
        dual_res = float('inf')
        epsilon_primal = 0
        epsilon_dual = 0
        epsilon_abs_term = math.sqrt(len(self.vars)) * self.epsilon_abs
        
        iteration = 0
        while (primal_res > epsilon_primal or dual_res > epsilon_dual) and iteration < self.max_iter:
            # Updates Lagrange multipliers and local copies
            for pot in self.pots.values():
                pot.update_lagrange_multipliers()
                pot.optimize_local_copies()

            primal_res = 0.0
            dual_res = 0.0
            ax_norm = 0.0
            bz_norm = 0.0
            ay_norm = 0.0
            
            # Updates variables, and computes dual residual and bz_norm
            for var in self.vars:
                total = 0.0
                for pot in self.var_locations[var]:
                    total += pot.get_total(var)

                new_value = total / len(self.var_locations[var])
                new_value = min(max(new_value, 0), 1)
                dual_res += (var.value - new_value) ** 2 * len(self.var_locations[var])
                bz_norm += new_value ** 2 * len(self.var_locations[var])
                var.value = new_value

            # Computes primal residual, ax_norm, and ay_norm
            for pot in self.pots.values():
                primal_res += pot.get_primal_res()
                ax_norm += pot.get_ax_norm()
                ay_norm += pot.get_ay_norm()

            # Finalizes quantities for stopping criteria
            primal_res = math.sqrt(primal_res)
            dual_res = self.eta * math.sqrt(dual_res)
            epsilon_primal = epsilon_abs_term + self.epsilon_rel * max(math.sqrt(ax_norm), math.sqrt(bz_norm))
            epsilon_dual = epsilon_abs_term + self.epsilon_rel * math.sqrt(ay_norm)
            
            iteration += 1

            if iteration % 25 == 0:
                self.logger.debug('Completed ' + str(iteration) + ' iterations.')
                self.logger.debug('Primal Residual:\t' + "{:0.5}".format(primal_res) + '  \t\tDual Residual:\t' +
                                  "{:0.5}".format(dual_res))
                self.logger.debug('Epsilon Primal:\t' + "{:0.5}".format(epsilon_primal) + '  \t\tEpsilon Dual:\t' +
                                  "{:0.5}".format(epsilon_dual))

        self.logger.info('Finished optimization in ' + str(iteration) + ' iterations.')
        self.logger.info('Primal residual: ' + "{:0.4}".format(primal_res) +
                         '   Dual residual: ' + "{:0.4}".format(dual_res))

    def _init(self):
        self.var_locations = {}

        for var in self.vars:
            self.var_locations[var] = set()

        for pot in self.pots.values():
            for var in pot.get_vars():
                self.var_locations[var].add(pot)

        self.needs_init = False


class _ADMMPotential:
    
    def __init__(self, admm, weight):
        self.admm = admm
        self.weight = weight

    def update_lagrange_multipliers(self):
        raise NotImplementedError('This class is abstract.')

    def optimize_local_copies(self):
        raise NotImplementedError('This class is abstract.')

    def get_vars(self):
        raise NotImplementedError('This class is abstract.')

    def get_total(self, var):
        raise NotImplementedError('This class is abstract.')

    def get_ax_norm(self):
        raise NotImplementedError('This class is abstract.')

    def get_ay_norm(self):
        raise NotImplementedError('This class is abstract.')

    def get_primal_res(self):
        raise NotImplementedError('This class is abstract.')

    def __contains__(self, item):
        raise NotImplementedError('This class is abstract.')


class _ADMMOneVarBowlPotential(_ADMMPotential):

    def __init__(self, admm, weight, coeff, var, const):
        super(_ADMMOneVarBowlPotential, self).__init__(admm, weight)
        self.coeff = coeff
        self.var = var
        self.const = const

        self.local_copy = var.value
        self.lagrange = 0.0

    def update_lagrange_multipliers(self):
        self.lagrange = self.lagrange + self.admm.eta * (self.local_copy - self.var.value)

    def optimize_local_copies(self):
        self.local_copy = self.admm.eta * self.var.value - self.lagrange
        self.local_copy -= 2 * self.weight * self.coeff * self.const
        self.local_copy /= 2 * self.weight * self.coeff * self.coeff + self.admm.eta

    def get_vars(self):
        return {self.var}

    def get_total(self, var):
        return self.local_copy + self.lagrange / self.admm.eta

    def get_ax_norm(self):
        return self.local_copy ** 2

    def get_ay_norm(self):
        return self.lagrange ** 2

    def get_primal_res(self):
        return (self.var.value - self.local_copy) ** 2

    def __contains__(self, item):
        return self.var == item


class _ADMMOneVarHingePotential(_ADMMOneVarBowlPotential):

    def optimize_local_copies(self):
        self.local_copy = self.var.value - self.lagrange / self.admm.eta
        if self.coeff * self.local_copy + self.const > 0:
            super(_ADMMOneVarHingePotential, self).optimize_local_copies()


class _ADMMTwoVarBowlPotential(_ADMMPotential):

    def __init__(self, admm, weight, coeff1, var1, coeff2, var2, const):
        super(_ADMMTwoVarBowlPotential, self).__init__(admm, weight)
        self.coeff1 = coeff1
        self.var1 = var1
        self.coeff2 = coeff2
        self.var2 = var2
        self.const = const

        self.local_copy1 = var1.value
        self.local_copy2 = var2.value
        self.lagrange1 = 0.0
        self.lagrange2 = 0.0

    def update_lagrange_multipliers(self):
        self.lagrange1 = self.lagrange1 + self.admm.eta * (self.local_copy1 - self.var1.value)
        self.lagrange2 = self.lagrange2 + self.admm.eta * (self.local_copy2 - self.var2.value)

    def optimize_local_copies(self):
        self.local_copy1 = self.admm.eta * self.var1.value - self.lagrange1
        self.local_copy1 -= 2 * self.weight * self.coeff1 * self.const
        self.local_copy2 = self.admm.eta * self.var2.value - self.lagrange2
        self.local_copy2 -= 2 * self.weight * self.coeff2 * self.const

        a0 = 2 * self.weight * self.coeff1 * self.coeff1 + self.admm.eta
        b1 = 2 * self.weight * self.coeff2 * self.coeff2 + self.admm.eta
        a1b0 = 2 * self.weight * self.coeff1 * self.coeff2

        self.local_copy2 -= a1b0 * self.local_copy1 / a0
        self.local_copy2 /= b1 - a1b0 * a1b0 / a0

        self.local_copy1 -= a1b0 * self.local_copy2
        self.local_copy1 /= a0

    def get_vars(self):
        return {self.var1, self.var2}

    def get_total(self, var):
        if self.var1 == var:
            return self.local_copy1 + self.lagrange1 / self.admm.eta
        else:
            return self.local_copy2 + self.lagrange2 / self.admm.eta

    def get_ax_norm(self):
        ax_norm = self.local_copy1 ** 2
        ax_norm += self.local_copy2 ** 2
        return ax_norm

    def get_ay_norm(self):
        ay_norm = self.lagrange1 ** 2
        ay_norm += self.lagrange2 ** 2
        return ay_norm

    def get_primal_res(self):
        primal_res = (self.var1.value - self.local_copy1) ** 2
        primal_res += (self.var2.value - self.local_copy2) ** 2
        return primal_res

    def __contains__(self, item):
        return self.var1 == item or self.var2 == item


class _ADMMTwoVarHingePotential(_ADMMTwoVarBowlPotential):

    def optimize_local_copies(self):
        self.local_copy1 = self.var1.value - self.lagrange1 / self.admm.eta
        self.local_copy2 = self.var2.value - self.lagrange2 / self.admm.eta

        if self.coeff1 * self.local_copy1 + self.coeff2 * self.local_copy2 + self.const > 0:
            super(_ADMMTwoVarHingePotential, self).optimize_local_copies()
