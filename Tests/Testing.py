
import unittest
import numpy as np
import torch

from scipy.optimize import minimize
import scipy.integrate as integrate

import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import beta, uniform




#################
#### LIBRARY ####
#################

# ------------------------------

# SAMPLING

def CDF_Inverse(cdf_inv, n_samples):
    u = np.random.uniform(0, 1, n_samples)
    return cdf_inv(u)

def Rejection(f, g, inv_cdf, START, END, K, N):
    x = CDF_Inverse(inv_cdf, N)
    y = np.random.uniform(0, 1, N) * K * g(x)
    return x[y <= f(x)]

def optimize_envelope(f, START, END):
    def objective(params):
        mu, sigma = params
        x = np.linspace(START, END, 1000)
        g = norm.pdf(x, mu, sigma)
        K = np.max(f(x) / g)
        return K  

    init_params = [(START + END) / 2, (END - START) / 4]
    bounds = [(START, END), (1e-5, (END - START) / 2)]
    result = minimize(objective, init_params, bounds=bounds, method='Nelder-Mead')
    
    if not result.success:
        raise ValueError("Optimización no exitosa!")
    
    mu_opt, sigma_opt = result.x
    K_opt = result.fun
    
    return mu_opt, sigma_opt, K_opt


def direct_sampling(function, a, b, iters=1000):
    x = np.random.uniform(a, b, iters)
    y = function(x)
    estimate = np.mean(y) * (b - a)
    return estimate

def control_variates(function, a, b, iters=1000):
    x = np.random.uniform(0, 1, iters)
    y = function(x)
    c = - np.cov(y, x)[0, 1] / np.var(x)
    estimate = np.mean(y + c*(x - 0.5)) 
    return estimate

def importance_sampling(function, a, b, iters=1000):
    x_samples = np.linspace(a, b, iters)
    y_samples = function(x_samples)
    
    mu = np.mean(y_samples)
    std = np.std(y_samples)
    
    x = []
    while len(x) < iters:
        _x = np.random.normal(mu, std)
        if _x >= a and _x <= b:
            x.append(_x)
    x = np.array(x)

    y = function(x) / norm.pdf(x, mu, std)
    estimate = np.mean(y) * (b - a)
            
    return estimate

def stratified_sampling(function, a, b, iters=1000):
    strata = np.linspace(a, b, iters + 1)
    x = np.random.uniform(strata[:-1], strata[1:])
    y = function(x)
    estimate = np.mean(y)
    return estimate

def antithetic_sampling(function, a, b, iters=1000):
    u = np.random.uniform(a, b/2, int(iters / 2))
    x = np.append(u, (b - u))
    y = function(x)
    estimate = np.mean(y)
    return estimate

# HAMILTONIAN DYNAMICS

def symplectic_euler(initial_q, initial_p, D, dHdq, dHdp, steps=10000, dt=0.01):
    q = np.zeros((steps, D))
    p = np.zeros((steps, D))
    q[0], p[0] = initial_q, initial_p
    for i in range(steps-1):
        for d in range(D):
            p[i+1][d] = p[i][d] - dt*dHdq[d](q[i][d])
            q[i+1][d] = q[i][d] + dt*dHdp[d](p[i+1][d])
    return np.array(q), np.array(p)

def stormer_verlet(initial_q, initial_p, D, dHdq, dHdp, steps=10000, dt=0.01):
    # p(t+1/2) = p(t) - \frac{\Delta t}{2} \nabla_q V(q(t)) \\
	#	q(t+1) = q(t) + \Delta t M^{-1} p(t+1/2) \\
	#	p(t+1) = p(t+1/2) - \frac{\Delta t}{2} \nabla_q V(q(t+1))
    q = np.zeros((steps*2, D))
    p = np.zeros((steps*2, D))
    q[0], p[0] = initial_q, initial_p
    for t in range(0, 2*steps-2, 2):
        for d in range(D):
            p[t+1] = p[t] - dt/2*dHdq(q[t])
            q[t+2] = q[t] + dt*dHdp(p[t+1])
            p[t+2] = p[t+1] - dt/2*dHdq(q[t+2])
    # Mask for taking the odd indexes
    mask = np.arange(steps*2) % 2 == 0
    return np.array(q)[mask], np.array(p)[mask]

  # Muller-Brown potential
def compute_Muller_potential(scale, x):
    A = (-200.0, -100.0, -170.0, 15.0)
    beta = (0.0, 0.0, 11.0, 0.6)
    alpha_gamma = (
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-6.5, -6.5]),
        x.new_tensor([0.7, 0.7]),
    )
    ab = (
        x.new_tensor([1.0, 0.0]),
        x.new_tensor([0.0, 0.5]),
        x.new_tensor([-0.5, 1.5]),
        x.new_tensor([-1.0, 1.0]),
    )
    U = 0
    for i in range(4):
        diff = x - ab[i]
        U = U + A[i] * torch.exp(
            torch.sum(alpha_gamma[i] * diff**2, -1) + beta[i] * torch.prod(diff, -1)
        )
    U = scale * U
    return U

def compute_gradient(scale, x):
    # Function to compute the gradient of the Müller-Brown potential
    U = compute_Muller_potential(scale, x)
    grad_U = torch.autograd.grad(U, x, create_graph=True)[0]
    return grad_U

def gibbs_sampling_with_potential(scale, num_samples, step_size=0.5):
    samples = np.zeros((num_samples, 2))
    x = np.random.normal(0, 1, 2)  # Initial guess for x and y
    x_tensor = torch.tensor(x, dtype=torch.float32)
    U_x = compute_Muller_potential(scale, x_tensor)

    for i in range(num_samples):
        # Sample y from the conditional distribution of y | x
        y = np.random.normal(x[1], step_size)
        x_y = np.array([x[0], y])
        x_y_tensor = torch.tensor(x_y, dtype=torch.float32)
        U_x_y = compute_Muller_potential(scale, x_y_tensor)

        # Accept or reject the new y based on the potential
        exponent = torch.clip(U_x - U_x_y, max=50, min=-50)
        if np.exp(exponent) > np.random.rand():
            x[1] = y
            U_x = U_x_y

        # Sample x from the conditional distribution of x | y
        x_new = np.random.normal(x[0], step_size)
        y_x = np.array([x_new, x[1]])
        y_x_tensor = torch.tensor(y_x, dtype=torch.float32)
        U_y_x = compute_Muller_potential(scale, y_x_tensor)

        # Accept or reject the new x based on the potential
        exponent = torch.clip(U_x_y - U_y_x, max=50, min=-50)
        if np.exp(exponent) > np.random.rand():
            x[0] = x_new
            U_x = U_y_x

        samples[i] = x

    return samples

# Defining the target PDF based on the Müller-Brown potential
def target_pdf_muller_brown(x, scale=1.0, temperature=1.0):
    potential = compute_Muller_potential(scale, x)
    return torch.exp(-potential / temperature)

# Proposal distribution function: Gaussian centered at the current parameter
def proposal_dist_gaussian(current_param, std_dev=0.1):
    proposed_param = current_param + torch.randn_like(current_param) * std_dev
    return proposed_param

# Metropolis-Hastings algorithm adapted for the Müller-Brown potential
def metropolis_hastings_muller_brown(target_pdf, proposal_dist, initial_param, iterations, temperature=40.0, scale=1.0):
    current_param = initial_param
    samples = [current_param.numpy()]
    for _ in range(iterations):
        proposed_param = proposal_dist(current_param)
        acceptance_probability = min(
            1,
            (target_pdf(proposed_param, scale, temperature) / target_pdf(current_param, scale, temperature)).item()
        )
        if torch.rand(1).item() < acceptance_probability:
            current_param = proposed_param
        samples.append(current_param.numpy())
    return np.array(samples)
    
def kl_divergence(p, q):
    # Adding a small constant for smoothing
    epsilon = 1e-10
    p_smoothed = p + epsilon
    q_smoothed = q + epsilon
    return np.sum(np.where(p_smoothed != 0, p_smoothed * np.log(p_smoothed / q_smoothed), 0))

def generate_grid(x1_min, x1_max, x2_min, x2_max, size=100):
    x1 = torch.linspace(x1_min, x1_max, size)
    x2 = torch.linspace(x2_min, x2_max, size)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="ij")
    grid = torch.stack([grid_x1, grid_x2], dim=-1)
    x = grid.reshape((-1, 2))
    return x

# Units and constants
G = 4 * np.pi**2  # AU^3 / (year^2 * solar_mass)
AU = 1.496e11  # meters
year = 3.154e7  # seconds
solar_mass = 1.989e30  # kg

class CelestialBody:
    def __init__(self, name, mass, pos, vel):
        self.name = name
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.momentum = self.mass * self.vel

    def compute_force(self, other_body):
        r = np.linalg.norm(self.pos - other_body.pos)
        f = -(G * self.mass * other_body.mass / r**3) * (self.pos - other_body.pos)
        return f

    def update_position(self, dt):
        self.pos += dt * self.momentum / self.mass

    def update_momentum(self, dt, force):
        self.momentum += dt * force
# ------------------------------




#################
##### TESTS #####
#################

# ------------------------------

class TestSampling(unittest.TestCase):
    
    # Test for CDF_Inverse
    def test_CDF_Inverse_Uniform(self):
        a, b = -10, 5
        cdf_inv_uniform = lambda x : a + (b-a)*x
        uniform_mean = (a+b)/2
        
        cdf_inv_uniform_samples = CDF_Inverse(cdf_inv_uniform, 10000)
        self.assertAlmostEqual(np.mean(cdf_inv_uniform_samples), uniform_mean, delta=0.1)
    
    def test_CDF_Inverse_Exponential(self):
        lamb = 1
        cdf_inv_exp = lambda x : np.log(1-x) / (-lamb)
        exp_mean = 1/lamb
        
        cdf_inv_exp_samples = CDF_Inverse(cdf_inv_exp, 10000)
        self.assertAlmostEqual(np.mean(cdf_inv_exp_samples), exp_mean, delta=0.1)
        
        
    # Test for Rejection
    def test_aceptance_rate(self):
        START = -10
        END = 10
        n_samples = 5000
        n_runs = 40

        # Compute 'area' once outside the function
        func = lambda x: 0.3 * np.exp(-0.2 * x**2) + 0.7 * np.exp(-0.2 * (x - 5)**2)

        area = integrate.quad(func, START, END)[0]

        def f(x):
            return func(x) / area

        # mean and variance of the target distribution
        mean_theoretical = integrate.quad(lambda x: x * f(x), START, END)[0]
        var_theoretical = integrate.quad(lambda x: (x - mean_theoretical)**2 * f(x), START, END)[0]

        mu_opt, sigma_opt, K_opt = optimize_envelope(f, START, END)

        def g(x):
            return stats.norm.pdf(x, mu_opt, sigma_opt)

        def inv_cdf(p):
            return stats.norm.ppf(p, mu_opt, sigma_opt)

        # Compute 'K' once
        x = np.linspace(START, END, 100)
        K = np.max(f(x) / g(x))

        # Run Rejection Sampling
        samples = Rejection(f, g, inv_cdf, START, END, K, 100000)
        
        computed_mean = np.mean(samples)
        computed_var = np.var(samples)
        
        self.assertAlmostEqual(computed_mean, mean_theoretical, delta=0.1)
        self.assertAlmostEqual(computed_var, var_theoretical, delta=0.1)
        
    
    # Law of large numbers
    def test_LLN(self):
        # Sampling of a variable X ~ N(3, 4)
        X = np.random.normal(3, 2, 100000)
        computed_mean = np.mean(X)
        theoretical_mean = 3
        self.assertAlmostEqual(computed_mean, theoretical_mean, delta=0.1)
    
    # Central limit theorem
    def test_CLT(self):
        # Sampling of a variable (X_i) with X_i ~ exp(1)
        X = np.random.exponential(1, (10000, 100))
        computed_mean = np.mean(X)
        theoretical_mean = 1
        self.assertAlmostEqual(computed_mean, theoretical_mean, delta=0.1)
        
    
    # Variance reduction techniques
    
    # Direct Sampling
    def test_direct_sampling(self):
        f = lambda x: x**2
        y = direct_sampling(f, 0, 1, 10000)
        self.assertAlmostEqual(y, 1/3, delta=0.1)
        
    # Control Variates
    def test_control_variates(self):
        f = lambda x: x**2
        y = control_variates(f, 0, 1, 10000)
        self.assertAlmostEqual(y, 1/3, delta=0.1)
        
        
    # Importance Sampling
    def test_importance_sampling(self):
        alfa, betax = 2.9, 1
        x = beta.rvs(alfa, betax, size=10000)
        y = x ** 2 / beta.pdf(x, alfa, betax)
        computed_mean = np.mean(y)
        self.assertAlmostEqual(computed_mean, 1/3, delta=0.1)
        
    # Stratified Sampling
    def test_stratified_sampling(self):
        f = lambda x: x**2
        y = stratified_sampling(f, 0, 1, 10000)
        self.assertAlmostEqual(y, 1/3, delta=0.1)
    
    # Antithetic Sampling
    def test_antithetic_sampling(self):
        f = lambda x: x**2
        y = antithetic_sampling(f, 0, 1, 10000)
        self.assertAlmostEqual(y, 1/3, delta=0.1)
    
    # Comparison of variance reduction techniques
    def test_variance_reduction(self):
        f = lambda x: x**2
        direct, control, impor, strat, anti = np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100)
        
        for i in range(100):
            direct[i] = direct_sampling(f, 0, 1, 1000)
            control[i] = control_variates(f, 0, 1, 1000)
            impor[i] = importance_sampling(f, 0, 1, 1000)
            strat[i] = stratified_sampling(f, 0, 1, 1000)
            anti[i] = antithetic_sampling(f, 0, 1, 1000)
        
        self.assertGreater(np.var(direct), np.var(anti))
        self.assertGreater(np.var(anti), np.var(control))
        self.assertGreater(np.var(impor), np.var(strat))
        

class TestHamiltonianDynamics(unittest.TestCase):

    def test_symplectic_euler_energy_conservation(self):
        # Harmonic oscillator parameters
        k = 1.0  # Spring constant
        m = 1.0  # Mass
        H = lambda q, p: 0.5 * k * q ** 2 + 0.5 * p ** 2 / m

        # Initial conditions
        initial_q = 1.0
        initial_p = 0.0
        steps = 10000
        dt = 0.01

        # Run symplectic euler
        q, p = symplectic_euler(initial_q, initial_p, 1, [lambda q: k * q], [lambda p: p / m], steps, dt)

        # Check energy conservation
        initial_energy = H(initial_q, initial_p)
        final_energy = H(q[-1], p[-1])
        self.assertAlmostEqual(initial_energy, final_energy, delta=0.1)

    def test_symplectic_euler_output_dimension(self):
        # Parameters
        steps = 10000
        dt = 0.01
        initial_q = 1.0
        initial_p = 0.0

        # Run symplectic euler
        q, p = symplectic_euler(initial_q, initial_p, 1, [lambda q: q], [lambda p: p], steps, dt)

        # Check output dimension
        self.assertEqual(q.shape, (steps, 1))
        self.assertEqual(p.shape, (steps, 1))
      
    def test_stormer_verlet_energy_conservation(self):
        # Harmonic oscillator parameters
        k = 1.0  # Spring constant
        m = 1.0  # Mass
        H = lambda q, p: 0.5 * k * q ** 2 + 0.5 * p ** 2 / m

        # Initial conditions
        initial_q = 1.0
        initial_p = 0.0
        steps = 10000
        dt = 0.01

        # Run Stormer-Verlet
        q, p = stormer_verlet(initial_q, initial_p, 1, lambda q: k * q, lambda p: p / m, steps, dt)

        # Check energy conservation
        initial_energy = H(initial_q, initial_p)
        final_energy = H(q[-1], p[-1])
        self.assertAlmostEqual(initial_energy, final_energy, delta=0.1)

    def test_stormer_verlet_output_dimension(self):
        # Parameters
        steps = 10000
        dt = 0.01
        initial_q = 1.0
        initial_p = 0.0

        # Run Stormer-Verlet
        q, p = stormer_verlet(initial_q, initial_p, 1, lambda q: q, lambda p: p, steps, dt)

        # Check output dimension
        self.assertEqual(q.shape, (steps, 1))
        self.assertEqual(p.shape, (steps, 1))

    def test_stormer_verlet_accuracy(self):
        # Parameters
        steps = 10000
        dt = 0.01
        initial_q = 1.0
        initial_p = 0.0
        t = np.arange(0, steps * dt, dt)

        # Exact solution for a simple harmonic oscillator
        q_exact, p_exact = np.cos(t), -np.sin(t)

        # Run Stormer-Verlet
        q, p = stormer_verlet(initial_q, initial_p, 1, lambda q: q, lambda p: p, steps, dt)

        # Check accuracy
        np.testing.assert_array_almost_equal(q.flatten(), q_exact, decimal=2)
        np.testing.assert_array_almost_equal(p.flatten(), p_exact, decimal=2)

class TestSolarSystemSimulation(unittest.TestCase):

    def setUp(self):
        # Initial setup for each test
        self.bodies = [
            CelestialBody("Sun", 1, [0, 0], [0, 0]),
            CelestialBody("Mercury", 1.6505e-7, [0, 0.39], [9.992, 0]),
            CelestialBody("Venus", 2.4335e-06, [0, 0.72], [7.38, 0]),
            CelestialBody("Earth", 2.986e-06, [0, 1], [6.282, 0]),
            CelestialBody("Mars", 3.2085e-07, [0, 1.52], [5.08, 0]),
            CelestialBody("Jupiter", 9.495e-04, [0, 5.187], [2.7615, 0]),
            # Add other celestial bodies as needed
        ]

    def test_energy_conservation(self):
        def symplectic_euler_step(bodies, dt):
            # Update positions
            for body in bodies:
                body.update_position(dt)

            # Update momenta
            for body in bodies:
                total_force = np.zeros(2)
                for other_body in bodies: 
                    if body != other_body:
                        total_force += body.compute_force(other_body)
                body.update_momentum(dt, total_force)

        def simulate_symplectic_euler(bodies, T, dt):
            times = np.arange(0, T, dt)
            positions = {body.name: [] for body in bodies}

            for t in times:
                for body in bodies:
                    positions[body.name].append(body.pos.copy())

                symplectic_euler_step(bodies, dt)

            return positions
        
        def gravitational_potential_at_point(point, body):
            r = np.linalg.norm(point - body.pos)
            if r == 0:  
                return 0
            return -G * body.mass / r
        
        def total_potential_at_point(point, bodies):
            total_potential = 0
            for body in bodies:
                total_potential += gravitational_potential_at_point(point, body)
            return total_potential
        
        def total_energy(bodies):
            kinetic = sum(0.5 * np.transpose(body.momentum).dot((np.identity(2) * 1 / body.mass).dot(body.momentum)) for body in bodies)
            potential = sum(body.mass * total_potential_at_point(body.pos, bodies) for body in bodies)
            return kinetic + potential
        
        initial_energy = total_energy(self.bodies)
        simulate_symplectic_euler(self.bodies, 1, 1/365)  # Simulate for 1 year
        final_energy = total_energy(self.bodies)

        self.assertAlmostEqual(initial_energy, final_energy, delta=1e-4)

class MCM(unittest.TestCase):
    def metropolis_hastings(self, target_pdf, proposal_dist, proposal_pdf, initial_param, iterations):
        current_param = initial_param
        samples = [current_param]
        
        for _ in range(iterations):
            proposed_param = proposal_dist(current_param)
            divi = target_pdf(current_param) * proposal_pdf(proposed_param, current_param) + 1e-10
            acceptance_probability = min(1, target_pdf(proposed_param) * proposal_pdf(current_param, proposed_param) /
                                        divi)
            
            if np.random.rand() < acceptance_probability:
                current_param = proposed_param
            
            samples.append(current_param)
        
        return samples

    # Example target distribution: Normal distribution
    def normal_pdf(self, x, mean=0, std_dev=1):
        return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * ((x - mean) / std_dev)**2)

    # Example proposal distribution and pdf: Normal distribution centered at current parameter
    def normal_proposal_dist(self, x, std_dev=1):
        return np.random.normal(x, std_dev)

    def normal_proposal_pdf(self, x, y, std_dev=1):
        return self.normal_pdf(y, mean=x, std_dev=std_dev)

    def test_convergence_to_target_distribution_metropolis(self):
        # Test if the samples converge to the target distribution
        iterations = 100000
        initial_param = 0
        alpha, beta = 2, 1
        gamma = lambda x: (beta**alpha / np.math.gamma(alpha)) * x**(alpha - 1) * np.exp(-beta * x)

        samples = self.metropolis_hastings(gamma, 
                                      self.normal_proposal_dist, 
                                      self.normal_proposal_pdf, 
                                      initial_param, 
                                      iterations)
        
        # Check if the mean and variance of the samples are close to that of the target distribution
        sample_mean = np.mean(samples)
        sample_variance = np.var(samples)
        
        target_mean = alpha / beta
        target_variance = alpha / beta**2

        self.assertAlmostEqual(sample_mean, target_mean, delta=0.1)
        self.assertAlmostEqual(sample_variance, target_variance, delta=0.2)
        
    
    # Gibbs Sampling
    def gibbs_sampling(self, mu, sigma, num_samples):
        samples = np.zeros((num_samples, 2))
        x = np.random.normal(0, 1)  # Initial guess for x

        for i in range(num_samples):
            # Sample y from the conditional distribution of y | x
            y_mean = mu[1] + sigma[1, 0] / sigma[0, 0] * (x - mu[0])
            y = np.random.normal(y_mean, np.sqrt(sigma[1, 1] - sigma[1, 0]**2 / sigma[0, 0]))
            samples[i, 1] = y

            # Sample x from the conditional distribution of x | y
            x_mean = mu[0] + sigma[0, 1] / sigma[1, 1] * (y - mu[1])
            x = np.random.normal(x_mean, np.sqrt(sigma[0, 0] - sigma[0, 1]**2 / sigma[1, 1]))
            samples[i, 0] = x

        return samples

    def test_convergence_to_target_distribution_gibbs(self):
        # Parameters for a bivariate normal distribution
        mu = [0, 0]  # Means
        sigma = np.array([[1, 0.5], [0.5, 1]])  # Covariance matrix

        # Generate samples
        num_samples = 100000
        samples = self.gibbs_sampling(mu, sigma, num_samples)

        # Check if the mean and covariance of the samples are close to that of the target distribution
        sampled_mean = np.mean(samples, axis=0)
        sampled_covariance = np.cov(samples.T)

        np.testing.assert_array_almost_equal(sampled_mean, mu, decimal=2)
        np.testing.assert_array_almost_equal(sampled_covariance, sigma, decimal=2)
        

class TestMullerBrownSampling(unittest.TestCase):

    def setUp(self):
        self.initial_param = torch.tensor(np.array([np.random.uniform(-1.5, 1.0), np.random.uniform(-0.5, 2)]), dtype=torch.float32)
        self.num_samples_max = 10000
        self.scale = 1.0
        self.grid_size = 100
        self.threshold = 1.0
        
        # Generate grid for theoretical probability distribution
        x_grid = generate_grid(-1.5, 1, -0.5, 2, self.grid_size)
        U = compute_Muller_potential(self.scale, x_grid).reshape(self.grid_size, self.grid_size)
        U = U.numpy()
        U[U > 100] = 100
        U /= U.max()
        self.prob_theoretical = np.exp(-U)
        self.prob_theoretical /= np.sum(self.prob_theoretical)
        
        # Generate samples using Gibbs and Metropolis-Hastings methods
        self.gibbs_samples = gibbs_sampling_with_potential(self.scale, self.num_samples_max)
        self.metropolis_samples = metropolis_hastings_muller_brown(target_pdf_muller_brown, proposal_dist_gaussian, self.initial_param, self.num_samples_max, temperature=40.0)

    def test_final_kl_divergence_below_threshold(self):

        # Create 2D histograms for the samples
        hist_gibbs, _, _ = np.histogram2d(self.gibbs_samples[:, 0], self.gibbs_samples[:, 1], bins=self.grid_size, range=[[-1.5, 1], [-0.5, 2]])
        hist_metropolis, _, _ = np.histogram2d(self.metropolis_samples[:, 0], self.metropolis_samples[:, 1], bins=self.grid_size, range=[[-1.5, 1], [-0.5, 2]])

        # Normalize to create probability distributions
        prob_gibbs = hist_gibbs / np.sum(hist_gibbs)
        prob_metropolis = hist_metropolis / np.sum(hist_metropolis)

        # Compute KL divergence for both methods
        kl_gibbs = kl_divergence(prob_gibbs, self.prob_theoretical)
        kl_metropolis = kl_divergence(prob_metropolis, self.prob_theoretical)
        
        self.assertLess(kl_gibbs, self.threshold, "Gibbs sampling KL divergence is above the threshold.")
        self.assertLess(kl_metropolis, self.threshold, "Metropolis-Hastings KL divergence is above the threshold.")
    
    
    def test_final_kl_divergence_below_threshold(self):

        gibbs_kls = []
        metropolis_kls = []

        for n in range(1, self.num_samples_max):
            # Create 2D histograms for the samples
            hist_gibbs, _, _ = np.histogram2d(self.gibbs_samples[:n, 0], self.gibbs_samples[:n, 1], bins=self.grid_size, range=[[-2, 1], [-0.5,2]])
            hist_metropolis, _, _ = np.histogram2d(self.metropolis_samples[:n, 0], self.metropolis_samples[:n, 1], bins=self.grid_size, range=[[-2,1], [-0.5, 2]])
            
            # Normalize to create probability distributions
            prob_gibbs = hist_gibbs / np.sum(hist_gibbs)
            prob_metropolis = hist_metropolis / np.sum(hist_metropolis)
            
            # Compute KL divergence for both methods
            gibbs_kls.append(kl_divergence(prob_gibbs, self.prob_theoretical))
            metropolis_kls.append(kl_divergence(prob_metropolis, self.prob_theoretical))
        
        self.assertGreater(np.mean(gibbs_kls[1:10]), gibbs_kls[-1], "Gibbs sampling KL divergence is not decreasing.")
        self.assertGreater(np.mean(gibbs_kls[90:100]), gibbs_kls[-1], "Gibbs sampling KL divergence is not decreasing.")
        self.assertGreater(np.mean(gibbs_kls[990:1000]), gibbs_kls[-1], "Gibbs sampling KL divergence is not decreasing.")
        
        self.assertGreater(np.mean(metropolis_kls[1:10]), metropolis_kls[-1], "Metropolis sampling KL divergence is not decreasing.")
        self.assertGreater(np.mean(metropolis_kls[90:100]), metropolis_kls[-1], "Metropolis sampling KL divergence is not decreasing.")
        self.assertGreater(np.mean(metropolis_kls[990:1000]), metropolis_kls[-1], "Metropolis sampling KL divergence is not decreasing.")

class TestMolecularSimulation(unittest.TestCase):

    def setUp(self):
        # Constants
        self.kT = 1.0  # Boltzmann constant times temperature
        self.epsilon = 1.0  # Depth of potential well
        self.sigma = 1.0  # Finite distance where potential is zero
        self.num_molecules = 30  # Number of molecules
        self.box_size = 10  # Size of the simulation box
        self.max_displacement = 0.1  # Maximum displacement in a single step

        # Initialize positions in a 2D lattice
        np.random.seed(0)
        self.positions = np.random.rand(self.num_molecules, 2) * self.box_size

    def lennard_jones_potential(self, r):
        """ Calculate Lennard-Jones potential for a distance r """
        r6 = (self.sigma / r)**6
        r12 = r6**2
        return 4 * self.epsilon * (r12 - r6)

    def total_potential_energy(self, positions):
        """ Compute the total potential energy of the system """
        energy = 0.0
        for i in range(self.num_molecules):
            for j in range(i+1, self.num_molecules):
                distance = np.linalg.norm(positions[i] - positions[j])
                energy += self.lennard_jones_potential(distance)
        return energy

    def metropolis_step(self, positions, total_energy):
        """ Perform one step of the Metropolis algorithm """
        # Choose a random molecule
        molecule_idx = np.random.randint(self.num_molecules)
        old_position = positions[molecule_idx].copy()

        # Move the molecule to a new position
        displacement = (np.random.rand(2) - 0.5) * self.max_displacement
        positions[molecule_idx] += displacement

        # Apply periodic boundary conditions
        positions[molecule_idx] %= self.box_size

        # Calculate the energy change
        new_energy = self.total_potential_energy(positions)
        delta_energy = new_energy - total_energy

        # Metropolis criterion
        if delta_energy > 0 and -delta_energy / self.kT < np.log(np.random.rand()):
            # Reject the move, revert to the old position
            positions[molecule_idx] = old_position
        else:
            # Accept the move, update total energy
            total_energy = new_energy

        return positions, total_energy

    def test_lennard_jones_potential(self):
        r = self.sigma
        potential = self.lennard_jones_potential(r)
        self.assertAlmostEqual(potential, 0.0)  # Lennard-Jones potential is zero at r = sigma

    def test_total_potential_energy(self):
        energy = self.total_potential_energy(self.positions)
        self.assertIsInstance(energy, float)

    def test_metropolis_step(self):
        total_energy = self.total_potential_energy(self.positions)
        positions = self.positions.copy()
        total_energies = [total_energy]
        for _ in range(500):
            positions, total_energy = self.metropolis_step(positions, total_energy)
            total_energies.append(total_energy)

        # Theoretical minimum energy for each pair is -epsilon
        min_energy = np.min(total_energies[100:])
        expected_min_energy = self.epsilon  # Three pairs, each contributing -epsilon
        self.assertGreater(min_energy, expected_min_energy)

if __name__ == '__main__':
    unittest.main()
