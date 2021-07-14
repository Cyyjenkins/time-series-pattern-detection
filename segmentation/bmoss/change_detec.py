# !/usr/bin/python

"""Bayesian model-based change detection for input-output sequence data

The Bayesian change-point detection model (BCDM) class implements a recursive
algorithm for partitioning a sequence of real-valued input-output data into
non-overlapping segments. The segment boundaries are chosen under the
assumption that, within each segment, the data follow a multi-variate linear
model.

Segmentation is carried out in an online fashion by recursively updating a set
of hypotheses. The hypotheses capture the belief about the current segment,
e.g. its duration and the linear relationship between inputs and outputs, given
all the data so far. Each time a new pair of data is received, the hypotheses
are propagated and re-weighted to reflect this new knowledge.

.. codeauthor:: Gabriel Agamennoni <gabriel.agamennoni@mavt.ethz.ch>
.. codeauthor:: Asher Bender <a.bender@acfr.usyd.edu.au>

"""
import numpy as np
from numpy import linalg
from numpy import random
from scipy import special


class MatrixVariateNormalInvGamma(object):
    """Matrix-variate normal, matrix-variate inverse gamma distribution

    The matrix-variate normal, inverse-gamma distribution is the conjugate
    prior for a matrix-variate normal distribution. As a result the
    distribution can be used in Bayesian estimation of the location and scale
    parameters of the matrix-variate normal distribution.

    """

    def __init__(self, mu, omega, sigma, eta):

        # Get size of data.
        m, n = np.shape(mu)
        self.__m, self.__n = m, n

        # Check that the location parameter is a matrix of finite numbers.
        if not (np.ndim(mu) == 2 and
                np.shape(mu) == (m, n) and
                not np.isnan(mu).any() and
                np.isfinite(mu).all()):
            msg = 'The location parameter must be a matrix of finite numbers.'
            raise Exception(msg)

        # Check that the scale parameter is a symmetric, positive-definite
        # matrix.
        if not (np.ndim(omega) == 2 and
                np.shape(omega) == (m, m) and
                not np.isnan(omega).any() and
                np.isfinite(omega).all() and
                np.allclose(np.transpose(omega), omega) and
                linalg.det(omega) > 0.0):
            msg = 'The scale parameter must be a symmetric, positive-definite'
            msg += ' matrix.'
            raise Exception(msg)

        # Check that the dispersion parameter is a symmetric, positive-definite
        # matrix.
        if not (np.ndim(sigma) == 2 and
                np.shape(sigma) == (n, n) and
                not np.isnan(sigma).any() and
                np.isfinite(sigma).all() and
                np.allclose(np.transpose(sigma), sigma) and
                linalg.det(sigma) > 0.0):
            msg = 'The noise parameter must be a symmetric, positive-definite'
            msg += ' matrix.'
            raise Exception(msg)

        # Check that the shape parameter is a number greater than one minus the
        # number of degrees of freedom.
        if not (np.isscalar(eta) and
                not np.isnan(eta) and
                np.isfinite(eta) and eta > n - 1.0):
            msg = 'The shape parameter must be greater than one minus the'
            msg += ' degrees of freedom.'
            raise Exception(msg)

        # Allocate space for storing the matrix of product statistics.
        self.__prod = np.zeros([m + n, m + n])

        # Initialize the statistics with the parameters of the prior
        # distribution.
        x = np.dot(omega, mu)
        self.__prod[:m, :m] = omega
        self.__prod[:m, m:] = x
        self.__prod[m:, :m] = x.T
        self.__prod[m:, m:] = np.dot(mu.T, x) + eta * sigma
        self.__weight = eta

    def update(self, X, Y):
        """Update the sufficient statistics given observed data.

        The sufficient statistics are the only parameters required to describe
        the shape of the distribution. Initially, the sufficient statistics
        contain no information apart from that implied by the prior
        distribution. As data arrive, the statistics are updated incrementally
        in order to reflect this new knowledge. Performing updates allows the
        sufficient statistics to summarise all information contained in the
        data observed so far.

        """

        # (Equation 5a, b)
        #
        #     | XX    XY |
        #     | YX    YY |
        #
        if np.ndim(X) > 1:
            k, m = np.shape(X)
            x = np.dot(X.T, Y)

            # Update the statistics given a block of data (in the following
            # order: XX, XY, YX, YY)
            self.__prod[:m, :m] += np.dot(X.T, X)
            self.__prod[:m, m:] += x
            self.__prod[m:, :m] += x.T
            self.__prod[m:, m:] += np.dot(Y.T, Y)
            self.__weight += k

        else:
            m = np.size(X)
            x = np.outer(X, Y)

            # Update the statistics given a single datum.
            self.__prod[:m, :m] += np.outer(X, X)
            self.__prod[:m, m:] += x
            self.__prod[m:, :m] += x.T
            self.__prod[m:, m:] += np.outer(Y, Y)
            self.__weight += 1

    def log_constant(self):

        m, n = self.__m, self.__n

        # Note usage of the log-determinant 'trick':
        #
        #     log(det(A)) = 2*sum(log(diag(chol(A))))
        #
        d = np.diag(linalg.cholesky(self.__prod))
        w = self.__weight

        # Evaluate the log-normalization constant.
        # (Equation 8)
        return special.gammaln(0.5*(w - np.arange(n))).sum() - \
               n * np.log(d[:m]).sum() - \
               w * np.log(d[m:] / np.sqrt(w)).sum() - \
               n * (0.5 * w) * np.log(0.5 * w)

    def parameters(self):
        """Return the posterior parameters.

        All the information content of the data is summarized by the sufficient
        statistics. As a result the posterior parameters are a function of the
        sufficient statistics. This is a consequence of the conjugacy of the
        matrix-variate Gaussian-inverse-Gamma distribution.

        """

        m = self.__m
        s = linalg.cholesky(self.__prod).transpose()
        w = self.__weight

        # Compute the parameters of the posterior distribution.
        return linalg.solve(s[:m, :m], s[:m, m:]), \
               np.dot(s[:m, :m].transpose(), s[:m, :m]), \
               np.dot(s[m:, m:].transpose(), s[m:, m:]) / w, \
               w

    def rand(self):

        m, n = self.__m, self.__n

        s = linalg.cholesky(self.__prod).transpose()
        w = self.__weight

        # Compute the parameters of the posterior distribution.
        mu = linalg.solve(s[:m, :m], s[:m, m:])
        omega = np.dot(s[:m, :m].transpose(), s[:m, :m])
        sigma = np.dot(s[m:, m:].transpose(), s[m:, m:]) / w
        eta = w

        # Simulate the marginal Wishart distribution.
        f = linalg.solve(np.diag(np.sqrt(2.0*random.gamma(
            (eta - np.arange(n))/2.0))) + np.tril(random.randn(n, n), -1),
                         np.sqrt(eta)*linalg.cholesky(sigma).transpose())
        b = np.dot(f.transpose(), f)

        # Simulate the conditional Gauss distribution.
        a = mu + linalg.solve(linalg.cholesky(omega).transpose(),
                              np.dot(random.randn(m, n),
                                     linalg.cholesky(b).transpose()))

        return a, b


class Bcdm(object):
    """Bayesian change detection model.

    Args:
        mu (numpy.array): (M x N) location parameters of the prior distribution.
        omega (numpy.array): (M x M) scale parameters of the prior distribution.
        sigma (numpy.array): (N x N) dispersion parameters of the prior distribution.
        eta (float): shape parameter of the prior distribution.
        alg (string): Specifies the algorithm to use. Choose either 'sumprod'
                      for the sum-product algorithm or 'maxprod' for the
                      max-product algorithm. If the sum-product algorithm is
                      selected, the posterior probabilities of the segmentation
                      hypotheses will be calculated. If the max-product
                      algorithm is selected, the most likely sequence
                      segmentation will be calculated.
        ratefun (float): Relative chance of a new segments being
                         generated. ``ratefun`` is a value between 0 and
                         1. Segments are MORE likely to be created with values
                         closer to zero. Segments are LESS likely to form with
                         values closer to 1. Alternatively, ratefun can be set
                         to an executable hazard function. The hazard function
                         must accept non-negative integers and return
                         non-negative floating-point numbers.
        basisfunc (callable): Feature functions for basis function
                              expansion. Feature functions provide additional
                              flexibility by mapping the predictor variables to
                              an intermmediate feature space, thus allowing the
                              user to model non-linear relationships.
        minprob (float): Minimum probability required for a
                         hypothesis. Hypotheses with insignificant support
                         (probabilities below this value) will be pruned.
        maxhypot (int): Maximum number of segmentation hypotheses to
                        consider. After each update, pruning will take place to
                        limit the number of hypotheses. If set to ``None``, no
                        pruning will NOT take place after updates, however,
                        pruning can be initiated manually by calling
                        :py:meth:`.trim`.

    Raises:
        Exception: If the any of the inputs are an incorrect type.

    """

    def __init__(self, mu=None, omega=None, sigma=None, eta=None,
                 alg='sumprod', ratefun=0.1, basisfunc=None, minprob=1.0e-6,
                 maxhypot=20):

        # The inference algorithm must be either sum-product or sum-product.
        if alg.lower() not in ['sumprod', 'maxprod']:
            msg = "The input 'alg' must be either 'sumprod' or 'maxprod'."
            raise Exception(msg)
        else:
            self.__alg__ = alg.lower()

        # Store number of dimensions in the predictor (independent/input
        # variable) and response (dependent/output variable) variables.
        self.__m = None
        self.__n = None

        # Allocate variables for matrix variate, normal inverse gamma
        # distributions.
        self.__mu = None
        self.__omega = None
        self.__sigma = None
        self.__eta = None

        # Set prior for the location parameter.
        if mu is not None:
            self.__mu = mu

        # Set prior for the scale parameter.
        if omega is not None:
            self.__omega = omega

        # Set prior for the dispersion/noise parameter.
        if sigma is not None:
            self.__sigma = sigma

        # Set prior for the shape parameter.
        if eta is not None:
            self.__eta = eta

        # Ensure algorithm initialises on first call to update.
        self.__initialised = False

        # If 'maxhypot' is set to none, no hypotheses will be trimmed.
        if maxhypot > 0 or not None:
            self.__maximum_hypotheses = maxhypot
        else:
            msg = "The input 'maxhypot' must be an integer greater than zero."
            raise Exception(msg)

        if minprob > 0:
            self.__minimum_probability = minprob
        else:
            msg = "The input 'minprob' must be a float greater than zero."
            raise Exception(msg)

        # Allocate variables for tracking segments.
        self.__hypotheses = list()
        self.__counts = list()
        self.__probabilities = list()

        # Store basis and hazard function.
        self.__basisfunc = basisfunc if callable(basisfunc) else lambda x: x
        self.__ratefun = ratefun if callable(ratefun) else lambda x: ratefun

    def __initialise_algorithm(self, m, n):
        """Initialise the Bcdm algorithm."""

        # Ensure input dimensions are consistent.
        if self.__m is None:
            self.__m = m
        elif self.__m != m:
            msg = 'Expected {} dimensions in the predictor variable.'.format(m)
            raise Exception(msg)

        # Ensure output dimensions are consistent.
        if self.__n is None:
            self.__n = n
        elif self.__n != n:
            msg = 'Expected {} dimensions in the response variable.'.format(n)
            raise Exception(msg)

        # Set uninformative prior for the location parameter.
        if self.__mu is None:
            self.__mu = np.zeros([m, n])

        # Set uninformative prior for the scale parameter.
        if self.__omega is None:
            self.__omega = np.eye(m)

        # Set uninformative prior for the dispersion/noise parameter.
        if self.__sigma is None:
            self.__sigma = np.eye(n)

        # Set uninformative prior for the shape parameter.
        if self.__eta is None:
            self.__eta = n

        # Create the initial hypothesis, which states that the first segment is
        # about to begin.
        self.__add_new_hypothesis(0.0)

    def __soft_max(self, x, y):
        return max(x, y) + np.log1p(np.exp(-abs(x - y)))

    def __add_new_hypothesis(self, log_likelihood, basisfunc=None):
        """Function for spawning new hypothesis"""

        # Set basis function.
        if basisfunc is None:
            basisfunc = self.__basisfunc

        # Create new Bayesian linear model (using supplied priors).
        stat = MatrixVariateNormalInvGamma(self.__mu,
                                           self.__omega,
                                           self.__sigma,
                                           self.__eta)

        # Add a new hypothesis, which states that a new segment is about to
        # begin.
        self.__hypotheses.append({'count': 0,
                                  'log_probability': log_likelihood,
                                  'distribution': stat,
                                  'log_constant': stat.log_constant(),
                                  'basisfunc': basisfunc})

    def update(self, X, Y, basisfunc=None):
        """Update model with a single observation.

        When new input-output data is available, the model can be updated using
        this method. As more and more data are collected, the number of
        hypotheses grows, increasing the computational complexity. By default
        hypotheses are pruned at the end of each update (see
        :py:meth:`.trim`.). To disable hypotheses trimming, initialise the
        class with ``maxhypot`` set to ``None``.

        Args:
            X (numpy.array): Observed (1 x M) input data (predictor variable).
            Y (numpy.array): Observed (1 x N) output data (response variable).

        """

        # Initialise algorithm on first call to update. This allows the
        # algorithm to configure itself to the size of the first input/output
        # data if no hyper-parameters have been specified.
        if not self.__initialised:
            init_basis = self.__basisfunc if basisfunc is None else basisfunc
            x = init_basis(X)
            m = x.shape[1] if np.ndim(x) > 1 else X.size
            n = Y.shape[1] if np.ndim(Y) > 1 else Y.size
            self.__initialise_algorithm(m, n)
            self.__initialised = True

        # Get size of data.
        k = X.shape[0] if np.ndim(X) > 1 else X.size
        m, n = self.__m, self.__n

        # Allocate variables for the dynamic programming pass.
        loglik = -np.inf
        logmax = -np.inf
        logsum = -np.inf
        ind = np.nan

        # Update hypotheses by updating each matrix variate, normal inverse
        # gamma distribution over the linear models.
        for hypothesis in self.__hypotheses:

            # Update the sufficient statistics.
            hypothesis['distribution'].update(hypothesis['basisfunc'](X), Y)

            # Compute the log-normalization constant after the update
            # (posterior parameter distribution).
            # (Equation 8)
            n_o = hypothesis['log_constant']
            n_k = hypothesis['log_constant'] = hypothesis['distribution'].log_constant()

            # Evaluate the log-density of the predictive distribution.
            # (Equation 16)
            log_density = n_k - n_o - k * (0.5 * m * n) * np.log(2.0 * np.pi)

            # Increment the counter.
            hypothesis['count'] += 1

            # Accumulate the log-likelihood of the data.
            # (Equation 17)
            hazard = self.__ratefun(hypothesis['count'])
            aux = np.log(hazard) + log_density + hypothesis['log_probability']
            loglik = self.__soft_max(loglik, aux)

            # Keep track of the highest, log-likelihood.
            if aux > logmax:
                logmax, ind = aux, hypothesis['count']

            # Update and accumulate the log-probabilities.
            hypothesis['log_probability'] += np.log1p(-hazard) + log_density
            logsum = self.__soft_max(logsum, hypothesis['log_probability'])

        # In the max-product algorithm, keep track of the most likely
        # hypotheses.
        if self.__alg__ == 'maxprod':
            loglik = logmax
            self.__counts.append(ind)

        # Add a new hypothesis, which states that the next segment is about to
        # begin.
        self.__add_new_hypothesis(loglik, basisfunc)

        # Normalize the hypotheses so that their probabilities sum to one.
        logsum = self.__soft_max(logsum, loglik)
        for hypothesis in self.__hypotheses:
            hypothesis['log_probability'] -= logsum

        # Automatically trim hypothesis on each update if requested.
        if self.__maximum_hypotheses is not None:
            self.trim_hypotheses(minprob=self.__minimum_probability,
                                 maxhypot=self.__maximum_hypotheses)

        # In the sum-product algorithm, keep track of the probabilities.
        if self.__alg__ == 'sumprod':
            iteration = list()
            for hypothesis in self.__hypotheses:
                iteration.append((hypothesis['count'],
                                  hypothesis['log_probability']))

            self.__probabilities.append(iteration)

    def trim_hypotheses(self, minprob=1.0e-6, maxhypot=20):
        """Prune hypotheses to limit computational complexity.

        The computational complexity of the algorithm can be managed by
        limiting the number of hypotheses maintained. This method limits the
        number of hypotheses maintained by the model by:

            1) Removing any hypotheses with a support (probability) less than
               ``minprob``.

            2) Preserving the first ``maxhypot`` likely hypotheses and
               discarding the rest.

        """

        # Skip pruning if less hypotheses exist than the maximum allowed.
        if len(self.__hypotheses) <= maxhypot:
            return

        # Sort the hypotheses in decreasing log probability order.
        self.__hypotheses.sort(key=lambda dct: -dct['log_probability'])

        # Store the indices of likely hypotheses.
        minprob = np.log(minprob)
        index = [i for i, hypot in enumerate(self.__hypotheses)
                 if hypot['log_probability'] > minprob]

        # Trim the hypotheses.
        index = index[:maxhypot] if len(index) >= maxhypot else index
        self.__hypotheses = [self.__hypotheses[i] for i in index]

        # NOTE: This final ordering can preserve the original order of the
        #       hypotheses. Interestingly, the algorithm specified in update
        #       does not require that the hypotheses be ordered! This sort can
        #       safely be ignored.
        # self.__hypotheses.sort(key=lambda dct: dct['index'])

        # Normalize the hypotheses so that their probabilities sum to one.
        logsum = -np.inf
        for hypot in self.__hypotheses:
            logsum = self.__soft_max(logsum, hypot['log_probability'])
        for hypot in self.__hypotheses:
            hypot['log_probability'] -= logsum

    def infer(self):
        """Return posterior probabilities OR sequence segmentation.

        If the MAX-PRODUCT algorithm is selected, this method returns the most
        likely sequence segmentation as a list of integers. Each integer in the
        list marks where a segment begins.

        If the SUM-PRODUCT algorithm is selected, this method returns the
        posterior probabilities of the segmentation hypotheses as a numpy
        array. Rows in the array represent hypotheses and columns in the array
        represent data points in the time-series.

        Returns:
            object: This method returns the inference results. In the case of
                    the MAX-PRODUCT algorithm, the method returns the most
                    likely segmentation. In the case of the SUM-PRODUCT
                    algorithm, this method returns the posterior probabilities
                    of the segmentation hypotheses.

        """

        # In the max-product algorithm, the most likely hypotheses are
        # tracked. Recover the most likely segment boundaries by performing a
        # back-trace.
        if self.__alg__ == 'maxprod':

            # Find the most likely hypothesis.
            max_hypothesis = max(self.__hypotheses,
                                 key=lambda dct: dct['log_probability'])

            # Find the best sequence segmentation given all the data so far.
            segment_boundaries = [len(self.__counts) - 1, ]
            index = segment_boundaries[0] - 1
            count = max_hypothesis['count'] - 1
            while index > 0:
                index -= count
                segment_boundaries.insert(0, index)
                count = self.__counts[index - 1]

            return segment_boundaries

        # In the sum-product algorithm, the segment probabilities are
        # tracked. Recover the segment probabilities by formatting the stored
        # history.
        else:
            k = len(self.__probabilities)
            segment_probabilities = np.zeros((k + 1, k + 1))
            segment_probabilities[0, 0] = 1.0

            # Update hypotheses probabilities.
            for i in range(len(self.__probabilities)):
                for (j, probability) in self.__probabilities[i]:
                    segment_probabilities[j, i + 1] = np.exp(probability)

            # A segment always occurs at the beginning of the dataset.
            segment_probabilities[0, 0] = 1.0

            return segment_probabilities
