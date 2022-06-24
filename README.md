# Sticking-coefficient
This set of `python` scripts implements one-dimensional exact factorization analysis using the reverse engineer approach to compute the nuclear phase, nuclear density, Time-dependent potential energy surface (TDPES), and the residual of the continuity equation <img src="https://render.githubusercontent.com/render/math?math=\nabla_{z}.J_{z} %2B \partial_{t}\rho_{z} = f(z,t)">.

## 1. Python Setup
To get this set of scripts up running on your available computational resources, make sure to have the below libraries installed on Python 3.6 or newer.

```
1. scipy.
2. matplotlib, pandas.
3. numpy, csv. 
```
## 2. Input files
All the input files for the system under study are in the Results directories. Rename the folder to `Results` in case it has a different name.

```
1. input_NRG.dat: initial inputs for the many-body time-dependent Schrodinger equation.
2. Energy_BO.dat: Born-Oppenheimer potential energy surface for the system's ground state.
3. NACT1.dat, NACT2.dat: First and second-order non-adiabatic coupling terms.
4. Re_Beta.dat, Im_Beta.dat: Real and Imaginary components of the many-body time-dependent Schrodinger equation. 
```

## 3. Running the scripts and output files
By running `python exact_fact_analysis.py,` you can generate a `sticking_exact.mp4` video displaying the TDPES, the nuclear density and phases, and the residual of the continuity function <img src="https://render.githubusercontent.com/render/math?math=f(z,t)">.     

In the `exact_fact_analysis.py` file we may have following options: 

1.  density() 
2.  all_kinetic_energy()
3.  tdpes()
4.  ani = SubplotAnimation(z_vec(), BOPE, density(), nuclear_phase(), tdpes(), continuity_eq(), time_step)
5.  ani.save('sticking_exact.mp4')

The script will run by default only for lines 4. and 5., while the other lines remain commented. This creates `density.dat` (nuclear density), 
`exact_data.dat` (all kinetic energies contribution), and `TDPES.dat` (Time-dependent potential energy surface) as output files.

Running `python plot_Tn.py` will create a figure with all kinetic energies contribution. The `plt_dynamic.py` is the class to generate the `.mp4` file and is part of the `exact_fact_analysis.py` script.




