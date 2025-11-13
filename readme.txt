Numba-Optimized Dipole Array Simulation (E-Field)
Phased arrays are highly relevant in signal transmission and reception as the antenna characteristic can be tuned by the control of the phase-difference between each individual antenna. when indexing the individual antennas continuously from 0 to N it results in two collimated beams which are emitted from the array in an angle Theta to both sides. With: 

sin(Theta) = Delta(Phi) lambda/(2pi n d)  with Delta(Phi): phase-difference between the individual emitters in rad n: refractive index, lambda: wavelength, and d: distance between the individual antennas. 

Phased arrays are commonly used in Radar technology, Radio Astronomy, but also in optics e.g. in plasmonic array structures.

Starting with a simple static dipole, this notebook simulates the electric field of a one-dimensional dipole array and computes field components on Cartesian planes (X–Z, X–Y, Y–Z). The core is Numba-optimized (last cell) with no np.cross/np.dot and the loop uses only scalar arithmetic with strictly typed, contiguous arrays. As a better readable version a non optimized simulation is also presented. Dipole near and intermediate fields are calculated but not relevant for a farfield simulation but important at short distances.

Physical Model (Brief)
- Point dipoles along the x-axis with preset spacing d = lambda0/2.
- Field decomposition:
  Near field ~ 1/rho^3: E_near ~ 3 (rhat · p) rhat - p
  Intermediate ~ 1/rho^2: E_mid ~ -i [(rhat · p) rhat - p]
  Far field ~ 1/rho: E_far ~ p - rhat (rhat · p)
  with rho = k * |r|, rhat = r / |r|.
  (Global factors such as 1/(4*pi*epsilon0) are omitted; focus is on relative intensities and interference.)
  

Numba Optimizations Used
- Scalar math instead of vector APIs: dotpr = rhatx*px + ... etc. No np.cross/np.dot.
- Efficient phases: exp(i*theta) = cos(theta) + i*sin(theta), avoiding np.exp with complex args in the loop.
- Contiguity & dtypes: All arrays float64 / complex128, enforced via np.ascontiguousarray.
- Parallelization: Outer loop over pixels with numba.prange.
- Cached intermediates: compute inv_rho, inv_rho2, inv_rho3 once—no powers in the loop.
- Caching: cache=True speeds up repeated kernel use.

Installation
1) (Recommended) Create a virtual environment.
2) Install dependencies:
   pip install -r requirements.txt
   
Key Parameters
- lambda_0 (wavelength) and k = 2*pi/lambda_0
- N (number of dipoles) and dipole_distance (default: lambda_0/2)
- p (dipole vector; here aligned with z via alpha = 90 degrees)
- Grids: x, y, z (default: 200 x 200 points per plane)
- Phase study: phase_step = np.arange(0, 2, 0.2); per dipole: phases = |idx| * n * piPlanes & Calls (examples)
- X–Z plane (y = 0) — active in the code:
    Ex = compute_field(X, zeros_XZ, Z, 0, positions, phases, p, k)
    Ez = compute_field(X, zeros_XZ, Z, 2, positions, phases, p, k)
    Ey = compute_field(X_Y, Y, zeros_XY, 1, positions, phases, p, k)

- Y–Z plane (x = 0) — prepared:
    Ex_YZ = compute_field(zeros_YZ, Y_Z, Z_Y, 0, positions, phases, p, k)

- X–Y plane (z = 0) — prepared:
    Ex_XY = compute_field(X_Y, Y, zeros_XY, 0, positions, phases, p, k)
    
Extensions ideas
- 2D/3D arrays: place dipoles on x–y or x–y–z lattices.
- Amplitude/phase apodization: custom phases and weights per dipole.

License & Citation
If you use this simulation in a publication, please cite this repository/notebook and the libraries used (NumPy, Numba, Matplotlib).
