import gpytorch
import numpy as np
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from loguru import logger


class GPModel(ApproximateGP):

    def __init__(self, inducing_points):
        """
        This method constructs a mean module, a kernel module, a variational distribution object,
        and a variational strategy object. This method should also be responsible for construting
        whatever other modules might be necessary.
        """
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """
        Args:
        - x: input data matrix of size (batch_size, data_dim).

        Returns:
        A MultivariateNormal with the prior mean and covariance evaluated at x.
        Specifically, the vector mu_x and matrix K_xx (batch_size, batch_size)
        representing the prior mean and covariance matrix of the gp.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

