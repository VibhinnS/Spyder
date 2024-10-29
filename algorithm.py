import numpy as np
from typing import List, Tuple, Callable

class CMAES:
    def __init__(
        self,
        objective_funcs: List[Callable],
        bounds: List[Tuple[float, float]],
        sigma0: float = 0.3,
        population_size: int = None,
        max_iterations: int = 100,
        tolerance: float = 1e-8
    ):
        """
        CMA-ES optimizer specifically designed for 3D IC parameter optimization.
        
        Parameters:
        -----------
        objective_funcs : List[Callable]
            List of objective functions [thermal_obj, electrical_obj, ...]
        bounds : List[Tuple[float, float]]
            Parameter bounds
        sigma0 : float
            Initial step size
        population_size : int
            Population size (if None, will be calculated based on dimension)
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        """
        self.objective_funcs = objective_funcs
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.n_objectives = len(objective_funcs)
        
        # Strategy parameters
        self.sigma = sigma0
        self.lambda_ = population_size if population_size else 4 + int(3 * np.log(self.n_dims))
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights ** 2)
        
        # Adaptation parameters
        self.cc = (4 + self.mueff/self.n_dims) / (self.n_dims + 4 + 2 * self.mueff/self.n_dims)
        self.cs = (self.mueff + 2) / (self.n_dims + self.mueff + 5)
        self.c1 = 2 / ((self.n_dims + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.n_dims + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1)/(self.n_dims + 1)) - 1) + self.cs
        
        # Dynamic state
        self.mean = self._initialize_mean()
        self.pc = np.zeros(self.n_dims)
        self.ps = np.zeros(self.n_dims)
        self.B = np.eye(self.n_dims)
        self.D = np.ones(self.n_dims)
        self.C = self.B.dot(np.diag(self.D ** 2)).dot(self.B.T)
        
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.best_solutions = []

    def _initialize_mean(self) -> np.ndarray:
        """Initialize mean vector within bounds"""
        return np.array([
            np.random.uniform(low, high) 
            for low, high in self.bounds
        ])

    def _bound_weights(self, x: np.ndarray) -> float:
        """Calculate boundary penalty"""
        penalty = 0
        for i, (lower, upper) in enumerate(self.bounds):
            if x[i] < lower:
                penalty += (lower - x[i]) ** 2
            elif x[i] > upper:
                penalty += (x[i] - upper) ** 2
        return penalty

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate population with all objective functions"""
        n_solutions = population.shape[0]
        fitness = np.zeros((n_solutions, self.n_objectives))
        
        for i in range(n_solutions):
            # Add boundary penalty to maintain feasibility
            penalty = self._bound_weights(population[i])
            for j, func in enumerate(self.objective_funcs):
                fitness[i, j] = func(population[i]) + penalty
                
        return fitness

    def _update_evolution_paths(self, y_w: np.ndarray):
        """Update evolution paths"""
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  self.B.dot(y_w / self.sigma)
        
        hsig = np.linalg.norm(self.ps) / \
               np.sqrt(1 - (1 - self.cs) ** (2 * self.iterations)) / \
               self.chi_n < 1.4 + 2 / (self.n_dims + 1)
        
        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * \
                  self.B.dot(self.D * y_w)

    def optimize(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run the optimization process.
        
        Returns:
        --------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (solution, fitness) pairs representing the Pareto front
        """
        self.chi_n = np.sqrt(self.n_dims) * (1 - 1/(4*self.n_dims) + 1/(21*self.n_dims**2))
        
        for self.iterations in range(self.max_iterations):
            # Generate new population
            arz = np.random.randn(self.lambda_, self.n_dims)
            population = self.mean + self.sigma * (self.B.dot(self.D.reshape(-1,1) * arz.T)).T
            
            # Evaluate population
            fitness = self._evaluate_population(population)
            
            # Sort by weighted sum of objectives (for simplicity)
            weighted_fitness = np.mean(fitness, axis=1)
            sorted_indices = np.argsort(weighted_fitness)
            
            # Calculate weighted mean
            y_w = np.zeros(self.n_dims)
            for k in range(self.mu):
                y_w += self.weights[k] * arz[sorted_indices[k]]
            
            # Update mean
            old_mean = self.mean.copy()
            self.mean += self.sigma * self.B.dot(self.D * y_w)
            
            # Update evolution paths and covariance matrix
            self._update_evolution_paths(y_w)
            
            # Adapt covariance matrix
            self.C = (1 - self.c1 - self.cmu) * self.C + \
                    self.c1 * (self.pc.reshape(-1,1).dot(self.pc.reshape(1,-1)) + \
                    (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            
            for k in range(self.mu):
                self.C += self.cmu * self.weights[k] * \
                         arz[sorted_indices[k]].reshape(-1,1).dot(arz[sorted_indices[k]].reshape(1,-1))
            
            # Update B and D
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.abs(self.D))
            
            # Update step size
            self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chi_n - 1) * self.cs / self.damps)
            
            # Store best solutions
            self.best_solutions = [
                (population[i], fitness[i])
                for i in sorted_indices[:self.mu]
            ]
            
            # Check convergence
            if np.all(np.abs(old_mean - self.mean) < self.tolerance):
                break
        
        return self.best_solutions

    def get_best_compromise(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the solution with the best trade-off between objectives"""
        if not self.best_solutions:
            return None
            
        solutions, fitness = zip(*self.best_solutions)
        fitness_array = np.array(fitness)
        
        # Normalize objectives
        normalized_fitness = (fitness_array - fitness_array.min(axis=0)) / (
            fitness_array.max(axis=0) - fitness_array.min(axis=0)
        )
        
        # Use weighted sum method
        weights = np.ones(self.n_objectives) / self.n_objectives
        weighted_sums = normalized_fitness.dot(weights)
        best_idx = np.argmin(weighted_sums)
        
        return self.best_solutions[best_idx]