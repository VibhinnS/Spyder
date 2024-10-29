import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass

@dataclass
class Material:
    """Material properties for electrical and thermal simulation"""
    resistivity_0: float  # Base resistivity at 20°C (Ω⋅m)
    temp_coefficient: float  # Temperature coefficient of resistance (1/K)
    thermal_conductivity: float  # Thermal conductivity (W/m⋅K)
    
@dataclass
class FluidProperties:
    """Properties for fluid cooling simulation"""
    density: float  # kg/m³
    heat_capacity: float  # J/kg⋅K
    thermal_conductivity: float  # W/m⋅K
    velocity: float  # m/s

class PowerETSolver:
    def __init__(self, dx, dy, dz, material: Material):
        """
        Initialize the PowerET solver
        
        Args:
            dx, dy, dz: Grid spacing in x, y, z directions (can be non-uniform arrays)
            material: Material properties
        """
        self.dx = np.array(dx)
        self.dy = np.array(dy)
        self.dz = np.array(dz)
        self.nx = len(dx) + 1
        self.ny = len(dy) + 1
        self.nz = len(dz) + 1
        self.material = material
        
        # Initialize solution matrices
        self.voltage = np.zeros((self.nx, self.ny, self.nz))
        self.temperature = np.full((self.nx, self.ny, self.nz), 20.0)  # Start at 20°C
        self.current_density = np.zeros((self.nx, self.ny, self.nz, 3))
        
    def _build_electrical_matrix(self):
        """Build the sparse matrix for voltage distribution equation"""
        n = self.nx * self.ny * self.nz
        rows, cols, data = [], [], []
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    idx = self._get_index(i, j, k)
                    
                    # Get temperature-dependent resistivity
                    temp = self.temperature[i,j,k]
                    rho = self.material.resistivity_0 * (1 + self.material.temp_coefficient * (temp - 20))
                    
                    # Add diagonal term
                    rows.append(idx)
                    cols.append(idx)
                    diag = 0
                    
                    # Add neighbor terms (6-point stencil in 3D)
                    neighbors = [
                        (i-1,j,k), (i+1,j,k),
                        (i,j-1,k), (i,j+1,k),
                        (i,j,k-1), (i,j,k+1)
                    ]
                    
                    for ni, nj, nk in neighbors:
                        if (0 <= ni < self.nx and 
                            0 <= nj < self.ny and 
                            0 <= nk < self.nz):
                            
                            nidx = self._get_index(ni, nj, nk)
                            
                            # Calculate coefficient based on grid spacing
                            if ni != i:
                                coef = -1/(rho * self.dx[min(i,ni)])
                            elif nj != j:
                                coef = -1/(rho * self.dy[min(j,nj)])
                            else:
                                coef = -1/(rho * self.dz[min(k,nk)])
                                
                            rows.append(idx)
                            cols.append(nidx)
                            data.append(coef)
                            diag -= coef
                    
                    data[len(data)-len(neighbors):] = diag
                    
        return csr_matrix((data, (rows, cols)), shape=(n,n))
    
    def _build_thermal_matrix(self, fluid_properties: FluidProperties = None):
        """Build the sparse matrix for thermal equations"""
        n = self.nx * self.ny * self.nz
        rows, cols, data = [], [], []
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    idx = self._get_index(i, j, k)
                    
                    k_thermal = self.material.thermal_conductivity
                    
                    # Handle fluid properties if present
                    if fluid_properties and self._is_fluid_cell(i,j,k):
                        k_thermal = fluid_properties.thermal_conductivity
                    
                    rows.append(idx)
                    cols.append(idx)
                    diag = 0
                    
                    # Add neighbor terms
                    neighbors = [
                        (i-1,j,k), (i+1,j,k),
                        (i,j-1,k), (i,j+1,k),
                        (i,j,k-1), (i,j,k+1)
                    ]
                    
                    for ni, nj, nk in neighbors:
                        if (0 <= ni < self.nx and 
                            0 <= nj < self.ny and 
                            0 <= nk < self.nz):
                            
                            nidx = self._get_index(ni, nj, nk)
                            
                            # Calculate coefficient
                            if ni != i:
                                coef = -k_thermal/self.dx[min(i,ni)]
                            elif nj != j:
                                coef = -k_thermal/self.dy[min(j,nj)]
                            else:
                                coef = -k_thermal/self.dz[min(k,nk)]
                                
                            rows.append(idx)
                            cols.append(nidx)
                            data.append(coef)
                            diag -= coef
                            
                    # Add convection terms if applicable
                    if self._is_boundary_cell(i,j,k):
                        h = self._get_convection_coefficient(i,j,k)
                        if h > 0:
                            diag += h
                            
                    data[len(data)-len(neighbors):] = diag
                    
        return csr_matrix((data, (rows, cols)), shape=(n,n))
    
    def solve(self, voltage_bcs, heat_sources, fluid_properties=None,
             max_iterations=100, tolerance=1e-6):
        """
        Solve the coupled electrical-thermal problem
        
        Args:
            voltage_bcs: Dictionary of (i,j,k): voltage for boundary conditions
            heat_sources: Array of heat sources (W/m³)
            fluid_properties: Optional fluid properties for cooling
            max_iterations: Maximum number of coupling iterations
            tolerance: Convergence tolerance
        """
        prev_temp = np.copy(self.temperature)
        
        for iteration in range(max_iterations):
            # 1. Solve electrical problem
            A_elec = self._build_electrical_matrix()
            b_elec = self._build_voltage_rhs(voltage_bcs)
            v = spsolve(A_elec, b_elec)
            self.voltage = v.reshape((self.nx, self.ny, self.nz))
            
            # 2. Calculate Joule heating
            self._update_current_density()
            joule_heat = self._calculate_joule_heating()
            
            # 3. Solve thermal problem
            A_thermal = self._build_thermal_matrix(fluid_properties)
            b_thermal = self._build_thermal_rhs(heat_sources + joule_heat)
            t = spsolve(A_thermal, b_thermal)
            self.temperature = t.reshape((self.nx, self.ny, self.nz))
            
            # Check convergence
            temp_change = np.max(np.abs(self.temperature - prev_temp))
            if temp_change < tolerance:
                print(f"Converged after {iteration+1} iterations")
                break
                
            prev_temp = np.copy(self.temperature)
            
        return self.voltage, self.temperature, self.current_density
    
    def _get_index(self, i, j, k):
        """Convert 3D indices to 1D index"""
        return i + j*self.nx + k*self.nx*self.ny
    
    def _is_fluid_cell(self, i, j, k):
        """Override this method to define fluid regions"""
        return False
    
    def _is_boundary_cell(self, i, j, k):
        """Check if cell is on the boundary"""
        return (i == 0 or i == self.nx-1 or 
                j == 0 or j == self.ny-1 or 
                k == 0 or k == self.nz-1)
    
    def _get_convection_coefficient(self, i, j, k):
        """Override this method to define convection coefficients"""
        return 0.0
    
    def _update_current_density(self):
        """Calculate current density from voltage gradient"""
        dx = np.diff(self.voltage, axis=0)/self.dx[:,None,None]
        dy = np.diff(self.voltage, axis=1)/self.dy[None,:,None]
        dz = np.diff(self.voltage, axis=2)/self.dz[None,None,:]
        
        self.current_density[...,0] = -dx
        self.current_density[...,1] = -dy
        self.current_density[...,2] = -dz
    
    def _calculate_joule_heating(self):
        """Calculate Joule heating from current density"""
        return np.sum(self.current_density**2, axis=-1) * self.material.resistivity_0