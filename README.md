# kratos_galaxy

A Python toolkit for setting up and post-processing galaxy-scale hydrodynamic simulations run with the [Kratos](https://bitbucket.org/kratos-dev/kratos) MHD code. Handles initial condition generation, background gravitational potential computation, binary I/O, cooling and feedback table generation, and data visualization (including [yt](https://yt-project.org/) integration for interactive analysis and pyXsim for synthetic X-ray observations).

---

## Repository Structure

```
kratos_galaxy/
├── base/            # Core data structures, units, grid, profiles, and source models
├── init/            # Initial condition generation and binary I/O
├── visual/          # Post-processing, diagnostics, and visualization
├── tables/          # Cooling (CLOUDY) and feedback (Starburst99) table generation
├── notebooks/       # Jupyter notebooks and standalone analysis scripts
├── par_input/       # Kratos runtime parameter files for the MHD simulations
├── outputs/         # Generated data — ICs, cooling tables, feedback tables
└── README.md
```

---

## Modules

### `base/` — Core Utilities

The foundation of the toolkit. Provides a unit-aware numerical framework, coordinate grids, analytic density profiles, and element abundance data.

| File | Description |
|------|-------------|
| `unit.py` | Physical and astrophysical constants in CGS (h, kb, eV, c, G, mp, etc.). Defines code-unit conversions via user-specified `rho0`, `l0`, `t0`. Derives derived units: `m0` (mass), `v0` (velocity), `G_code` (G in code units). Exposes class properties for each constant. |
| `data_field.py` | Unit-aware array wrapper. Tracks [mass, length, time, temperature] dimensions through arithmetic, NumPy ufuncs, and array functions. Supports conversion between code units and CGS. Overrides operators (`+`, `-`, `*`, `/`, `**`, `@`, comparisons) with dimensional checking. Implements `__array_ufunc__` and `__array_function__` for seamless NumPy interop. |
| `grid.py` | N-dimensional grid object supporting Cartesian, cylindrical, and spherical coordinates. Provides meshgrid creation, coordinate transformations (cyl→sph, sph→cyl, cyl→cart, cart→cyl), logarithmic spacing, and `LinearNDInterpolator`-based data interpolation. Initialized from `.par` configs. |
| `profile_base.py` | Library of standard analytic density profiles — `disc`, `exp`, `gauss`, `uniform`, `sech2` — plus their derivatives (`rho_prime`). Each profile is a callable that takes cylindrical coordinates (R, z) and returns density. Used as building blocks for initial condition profiles. |
| `source.py` | Wraps an arbitrary density function as a `source` object with dimensional units (mass, length, time). Supports addition (`+`) and scalar multiplication (`*`) of source functions, enabling composition of complex background models from simpler components. |
| `elements.py` | Reference data: atomic number, atomic weight, and solar abundance (Asplund+2009) for elements H through Zn (30 elements). Used for metallicity-dependent physics. |
| `utils.py` | Utility: `config_setup(path)` — loads a `.par` config file via `configparser`, initializes the unit system from the `[unit]` section, and returns the parsed config object. Standard entry point for all scripts that read `.par` files. |

### `init/` — Initial Condition Generation

Generates initial conditions for Kratos galaxy simulations: density, velocity, metallicity, and gravitational potential on a 3D grid. Background potential can be computed analytically (for NFW halos and point sources) or numerically (for arbitrary density distributions via the FDM Poisson solver).

| File | Description |
|------|-------------|
| `init_gen.py` | Entry-point script. Parses a `.par` config and runs the full IC generation pipeline: `config_setup` → `profile` → `IC` → binary output. Accepts config file path as command-line argument. |
| `init.py` | Defines the `profile` class — builds 3D density and azimuthal velocity grids on a cylindrical coordinate system. Supports `zsolve` (vertical hydrostatic equilibrium solver with a Jeans-based cutoff) and `usr_func` (explicit function-based profiles). Also defines the `IC` class, which writes all IC data (density, velocity, metallicity, potential) to a Kratos-format binary file. Includes `gen_profiles_bin` for serializing profile metadata. |
| `background.py` | Defines the `background` class — compiles all stellar and dark matter source functions from the `.par` config into a self-consistent gravitational background. Each source is a callable density function ρ(R, z). Components can be toggled on/off in the config. Also defines `background_source` — wraps a source with total mass computed by 2D numerical integration (trapezoidal rule). Includes implementations for multiple published galactic component models (see below). |
| `FDM.py` | Finite-difference Poisson solver (`background_potential`). Computes the gravitational potential Φ(R, z) on a 2D spherical-polar grid by solving ∇²Φ = 4πGρ with Neumann (inner) and Dirichlet (outer) boundary conditions. Used when analytic-only potentials are insufficient (e.g., for complex bar+spiral models). Features adaptive boundary placement while preserving second-order accuracy. |
| `binary_io.py` | Low-level binary reader/writer for Kratos' native format. Endian-aware, handles header map parsing, variable-size binary fields, and streaming I/O. Supports chunked reading for large files. Adapts to Kratos' `output_param` format changes (Version 4+). |
| `setup.par` | Default IC config: Sormani+19 components (bulge, disc, bar), 512×256 FDM grid, disc+zsolve profiles. Outputs to `cmz_init.bin`. |
| `s19.par` | Variant of `setup.par`. Outputs to `cmz_init_s19.bin`. |
| `H24.par` | Alternate source model: Portail bar (`bar_portail`), nuclear stellar cluster (`nsc_s20`), nuclear stellar disc (`nsd_s20`), and `disc_h24`. Uses `usr_func` profile flag instead of `disc zsolve`. Outputs to `cmz_init_H24.bin`. |
| `setup_test.ipynb` | Jupyter notebook for interactively testing IC generation. Imports all `init/` and `base/` modules. |

#### Background Source Models

The `background` class in `background.py` contains density functions ρ(R, z) from the following published models. Each model parameterises a Galactic component (bulge, disc, bar, etc.) and is a callable of form `model(R, z, **params)`:

| Function | Component | Reference |
|----------|-----------|-----------|
| `disc_s19` | Nuclear gas disc | Sormani et al. 2019, MNRAS 488, 4663 |
| `bulge_s19` | Nuclear stellar bulge | Sormani et al. 2019, MNRAS 488, 4663 |
| `bar_s19` | Stellar bar (COBE/DIRBE fit) | Sormani et al. 2019, MNRAS 488, 4663 |
| `nsd_s20` | Nuclear stellar disc | Sormani et al. 2020, MNRAS 497, 5024 |
| `nsc_s20` | Nuclear stellar cluster | Sormani et al. 2020, MNRAS 497, 5024 |
| `disc_s22` | Stellar disc (young+old) | Sormani et al. 2022, MNRAS 512, 1857 |
| `bar_portail` | 3D stellar bar (made-to-measure) | Portail et al. 2017, MNRAS 465, 1621 |
| `disc_m17` | Thin + thick stellar discs | McMillan 2017, MNRAS 465, 76 |
| `disc_h24` | Exponential stellar disc | Custom (H24 model) |
| `halo` | NFW dark matter halo | Analytic; parameters in `[background_analytic]` |

### `visual/` — Post-Processing and Visualization

The primary interface for analysing simulation outputs. Provides data loading, field enrolment, diagnostics, and publication-quality plotting.

| File | Description |
|------|-------------|
| `proxy.py` | Convenience import hub for Jupyter notebooks. Imports all visual tools, enrols physical constants from the unit system, sets Matplotlib defaults (font, colourmap, font sizes), and patches NumPy print options. **Note:** unconditionally imports `pyxsim` — all consumers must have pyxsim installed. |
| `yt_kratos.py` | Bridges Kratos binary outputs to yt. `kratos_to_hdf5()` constructs a temporary `.athdf` (Athena++ HDF5) file from in-memory Kratos data. `load_kratos_yt()` builds the HDF5 file and returns a `yt.Dataset`, enabling yt's full analysis suite (projections, slices, phase plots, etc.). Supports extra derived fields and isothermal EOS variants. |
| `hydro_data_gal.py` | Extends the base `hydro_data` reader with galaxy-simulation-specific fields. Key capabilities: enrolment of metallicity (`[Z/H]`), entropy-based pressure, temperature (constant or variable μ), velocity, Mach number, azimuthal velocity, and virial parameter. Mesh-stitching (`stitch_fields`, `refined_stitch`) for combining parent+refined data. CIC particle-to-mesh mapping for spatially filtering sink particles. Particle spatial filtering with configurable coordinate cuts. |
| `plot_gal.py` | High-level plotting library (~1200 lines): `plot_lvd` — position-velocity longitude-velocity diagrams with Sun-centred ray-tracing (slab intersection path lengths). `f_plot_grid` — multi-panel field slice plots. `time_evo`/`plt_time_evo` — time series extraction and plotting across simulation outputs. `plt_snaps`/`plt_snaps_yt` — snapshot grids with shared colour bars. `plt_fields`/`plt_fields_yt` — multi-field single-snapshot panels. `plt_phase` — 2D phase diagrams (ρ–T) with mass-weighted histogram2d. Also includes `fstring` (zero-padded output numbering) and `figgen` (sequential figure factory). |
| `blk_func.py` | Per-block hydro analysis functions operating on individual mesh blocks. `bf_hyd_cmzmass` — gas mass within a cylinder. `bf_hyd_cmzvel` — mass-weighted azimuthal velocity in a cylinder. `bf_hyd_massflux` — vertical mass flux through a plane with cylindrical radial cut. `bf_hyd_cylflux` — lateral mass flux through a cylinder with exact circle-face intersection geometry. Each has a corresponding `f_blk_*` aggregator that sums over all blocks. |
| `par_func.py` | Particle-based quantity extractors: `f_par_SFR` — star formation rate from sink particles. `f_par_Mstar` — total stellar mass. `f_par_Nsne` — cumulative supernova count. `f_par_Mgas` — gas mass from particle mapping. |
| `diagnostics.py` | High-level diagnostic wrappers that combine block-level and particle-level quantities. `gas_mass()`, `mean_azimuthal_velocity()`, `mass_flux_zplane()`, `cylinder_mass_flux()`, `sfr()`, `stellar_mass()`, `n_supernovae()`, `gas_velocity_dispersion()`, `run_all()` (bulk snapshot diagnostics dictionary). |
| `enroll_virial.py` | Enrols the local virial parameter α_vir = σ_v² / (4πG ρ ℓ²) on every mesh block. Uses a 3-point stencil for velocity dispersion, Jeans-length-based spatial scale, and ghost-cell boundary handling. |

#### Key Conventions

- **Sun position**: R = 8150 pc, φ = 28° CCW from negative x (208° from +x/bar axis), z = 25 pc. Used throughout `plot_lvd` and `xray_allsky.py`.
- **Reflection**: `reflect="yz"` maps the simulation's bar (+x) axis to the correct quadrant in standard Galactic coordinates by reflecting x → −x. Consistent across all sky-mapping scripts.
- **Slab intersection path length**: `plot_lvd` and `xray_map.py` use the slab-intersection method (computing exact path length through rectangular cells along the LOS) for column density and surface brightness. `xray_allsky.py` uses cell volume integration instead (see below).

### `tables/` — Cooling & Feedback Table Generation

Generates tabulated cooling/heating rates and stellar feedback quantities for use by the Kratos MHD code at runtime. The tables are 3D/4D interpolators binned in (temperature, density, metallicity) for cooling and (stellar population age) for feedback.

| File | Description |
|------|-------------|
| `table_gen.py` | Entry-point script. Accepts a `.par` config (defaults to `table_setup.par`). Loads or regenerates CLOUDY and Starburst99 data, builds interpolators, and writes `cooling_table.dat` and `fdbk_table.dat`. Uses pickle caching in `obj/` to avoid re-parsing large CLOUDY outputs. |
| `table_setup.par` | Config: CLOUDY and SB99 directory paths, interpolation grid ranges (T, n_H, [Z/H] for cooling; stellar age for feedback), output paths, and caching options. |
| `cooling_conversion.py` | CLOUDY output processor. Reads `.ovr` (overview/metadata), `.abn` (abundances), `.ion` (ionisation fractions), and `.cool` (cooling/heating rates) files. Computes mean molecular weight μ from ionisation fractions. Builds `LinearNDInterpolator` objects for net cooling rate (erg cm³ s⁻¹) and μ. Writes the final table in Kratos' config format. |
| `SB99_conversion.py` | Starburst99 output processor. Reads CGKZ3 table files (yields, SNR rates, power output, ionising photon quanta). Extracts SNe rate, wind mass/energy/momentum injection rates, and ionising photon rate as functions of stellar population age. Writes the feedback table. |
| `obj/` | Cached pickle files (`cloudy_data.pkl`, `int_data.pkl`) for faster table regeneration. |

### `notebooks/` — Analysis Scripts and Notebooks

Self-contained analysis routines and interactive Jupyter notebooks for paper-quality science plots.

| File | Description |
|------|-------------|
| `xray_allsky.py` | **All-sky X-ray Mollweide maps** using pyXsim CIE emissivity (0.6–1 keV band). Computes volume photon emissivity via `CIESourceModel("cloudy")`, then calculates total photon rate per unit solid angle as ε × V_cell / (4π) [photons s⁻¹ sr⁻¹] (isotropic CIE emission). Cells are histogrammed by Galactic (l, b) from the Sun position (8150 pc, 208°). Produces per-timestep Mollweide maps over outputs 30–65 for both b10 (starburst) and b60 (quiescent) runs. Smoothing via Gaussian filter (σ=1 pixel = 0.1°). cf. Predehl+2020 eROSITA X-ray bubbles. |
| `clump_mass_function.py` | **Clump mass function** comparison between b10 (SB) and b60 (QU) runs using `astrodendro` dendrogram analysis on face-on surface density projections (12 kpc box, 2048² pixels). Fits power-law slopes dN/dM ∝ M^−α over 10⁴–2×10⁶ Msun and overplots an M^−2 reference line. |
| `power_spectrum.py` | **3D turbulent velocity power spectrum** via FFT with radial binning. Exports function `power_spectrum(ds, R_cut, z_cut, N)` → (k_vals, P_k). Samples velocity on a uniform Cartesian grid within cylindrical cuts, subtracts bulk flow, computes 3D FFT, and bins the power spectrum radially in k-space. |
| `cmz_plot.ipynb` | Jupyter notebook: better-organized science plots. |
| `cmz_scratch.ipynb` | Jupyter notebook: scratch work and debugging. |
| `cmz_paper.ipynb` | Jupyter notebook: official paper figures (largest notebook, ~8400 lines). |

### `par_input/` — Kratos Runtime Parameter Files

Input files for the Kratos MHD code controlling mesh refinement, physics modules, boundary conditions, and runtime parameters.

| File | Description |
|------|-------------|
| `cmz_HIP.par` | Full Kratos config for CMZ simulation on HIP (CPU) devices. AMR: 128³ root → 3 levels of refinement (64³ effective block resolution). Physics: 300 Myr runtime, bar + background rampup (0–146 Myr), NFW halo (ρ_halo = 5.8×10⁻³ Msun pc⁻³, r_halo = 100 kpc), 4×10⁶ Msun SMBH, sink particles with Bondi accretion, SN feedback (Kimm+ Cen scheme), photoionisation, stellar winds. Cooling via tabulated CLOUDY rates. Turbulence driving disabled. |
| `cmz_CUDA.par` | Identical physics, mesh, refinement, and runtime to HIP config. Differs only in device settings: `max_streams = 256` for CUDA GPU execution. |

Both reference IC files from `outputs/init_cond/` and cooling tables from `outputs/therm_tables/`.

### `outputs/` — Generated Data

Binary and table files generated by the toolkit, not committed to version control (`.gitignore` excludes `*.bin`).

```
outputs/
├── init_cond/
│   ├── cmz_init.bin          # Default IC (Sormani+19, disc+zsolve)
│   ├── cmz_init_s19.bin      # Sormani+19 variant
│   └── cmz_init_H24.bin      # H24 model (Portail bar, NSC, NSD)
├── therm_tables/
│   ├── cooling_table.dat     # Tabulated CLOUDY cooling/heating rates
│   └── cooling_table2.dat    # Alternate cooling table
└── fdbk_tables/
    ├── fdbk_table.dat        # Tabulated Starburst99 feedback quantities
    └── alpha_acc.dat         # Bondi accretion alpha parameter table
```

---

## Configuration File Format (`.par`)

All scripts (IC generation, table generation, Kratos runtime) use INI-style configuration files parsed by Python's `configparser`. Sections and their roles:

| Section | Used by | Controls |
|---------|---------|----------|
| `[unit]` | init, tables | Base code units: `rho0` (g cm⁻³), `l0` (cm), `t0` (s). Defines all derived units. |
| `[output_grid]` | init | Grid type (`cylindrical`), coordinate ranges, spacing, resolution (Nr, Nz, Nphi). |
| `[IC_profile]` | init | Profile method per dimension (`exp`, `disc`, `zsolve`, `usr_func`), profile parameter file, output filename. |
| `[IC_hydro]` | init | Gas density normalisation (`rhobase`), mean molecular weight (`mu`), initial temperature (`Tdisc`). |
| `[FDM]` | init | Finite-difference Poisson solver: toggle (`FDM_on`), grid dimensions (2D spherical-polar), boundary conditions, solver path. |
| `[background_src]` | init | Toggle individual source components (e.g. `disc_s19 = 1`, `bar_s19 = 1`). |
| `[background_mass]` | init | Optionally specify known component masses (Msun) to skip numerical integration. |
| `[background_analytic]` | init | NFW halo parameters (`rho_halo`, `r_halo`) and black hole mass (`M_bh`). |
| `[zsolve]` | init | Cutoff radii R_cut for the vertical hydrostatic equilibrium solver (per-zone constraining limits). |

---

## Quickstart

### Generating Initial Conditions

```bash
cd kratos_galaxy/init/
python init_gen.py setup.par
```

Output: `outputs/init_cond/cmz_init.bin`.

**Programmatic usage:**

```python
import configparser
from unit import units
from data_field import data_field
from init import profile, IC

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read("setup.par")
units.read_config(config)
data_field.unit_system = units()

prof = profile(config_flag='output_grid')
IC(prof)
```

### Generating Cooling and Feedback Tables

```bash
cd kratos_galaxy/tables/
python table_gen.py table_setup.par
```

Requires existing CLOUDY output files in `../../cloudyoutputs/production/` and Starburst99 CGKZ3 output in `../../Starburst99/CGKZ3/Output/`. Outputs appear in `outputs/therm_tables/` and `outputs/fdbk_tables/`.

### Running the Kratos Simulation

```bash
# HIP (CPU):
cd ~/scratch/cmz/cmz_test/
mpirun -np N kratos_hip par_input/cmz_HIP.par

# CUDA (GPU):
mpirun -np M kratos_cuda par_input/cmz_CUDA.par
```

Output snapshots are written as `cmz_fid4_b10_XXXXX.bin` (starburst) and `cmz_fid4_b60_XXXXX.bin` (quiescent).

### Running X-ray All-Sky Maps

```bash
cd kratos_galaxy/notebooks/
python3 xray_allsky.py
```

Processes both b10_SB and b60_QU runs for outputs 30–65. Outputs: `../../cmz/paper_plots/snap_plots/xray_mollweide_{OUT}_{run_label}.png`. Produces 36 Mollweide maps per run (one per timestep).

**Creating MP4 movies from the output maps:**

```bash
cd ~/scratch/cmz/paper_plots/snap_plots/
# b10 starburst
ffmpeg -framerate 3 -pattern_type glob -i "xray_mollweide_*_b10_SB.png" \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p \
  xray_mollweide_b10_SB.mp4

# b60 quiescent
ffmpeg -framerate 3 -pattern_type glob -i "xray_mollweide_*_b60_QU.png" \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p \
  xray_mollweide_b60_QU.mp4
```

### Loading Data in a Jupyter Notebook

```python
from visual.proxy import *   # imports units, yt helpers, plot tools

# Load a Kratos output file (last timestep, enrol primitive variables)
d = get_last_dd("output_00010.bin", -1, enroll_prim=True, data_type=galaxy_data)
d.load_particles()

# Quick multi-panel slice plot
fig = f_plot_grid(d, flds=['rho', 'T_ent', 'met'], axes=[2, 1])

# Longitude-velocity diagram
fig = plot_lvd(d, lbins=np.linspace(-180, 180, 361),
               bbins=np.linspace(-90, 90, 181),
               vbins=np.linspace(-200, 200, 401))
```

### Loading with yt

```python
from visual.yt_kratos import load_kratos_yt
from hydro_data import get_dd

d  = get_dd("output_00010.bin")
ds = load_kratos_yt(d)

# yt slice
slc = ds.slice('z', 0)
slc.plot('density').show()

# yt projection
prj = yt.ProjectionPlot(ds, 'z', ('gas', 'density'), weight_field=('gas', 'density'))
prj.show()
```

### Computing Diagnostic Quantities

```python
from visual.diagnostics import run_all

d = get_last_dd("output_00010.bin", -1, enroll_prim=True, data_type=galaxy_data)
d.load_particles()

results = run_all(d, R_out=500.0, Z_out=500.0)
# Returns dict with: gas_mass, mean_vphi, mass_flux_zplane,
#   cylinder_mass_flux, sfr, stellar_mass, n_sne, velocity_dispersion
```

### Computing Turbulence Power Spectra

```python
from notebooks.power_spectrum import power_spectrum

k, P_k = power_spectrum(ds, R_cut=500, z_cut=140, N=128)
# k in [1/pc], P_k in [km²/s² · pc³]
```

### Enrolling Custom yt Fields

```python
def _temp_K(field, data):
    arr = data[("athena_pp", "temperature")]
    return ds.arr(arr.v, "K")

ds.add_field(("gas", "temperature_K"), function=_temp_K,
             units="K", sampling_type="cell", force_override=True)
```

---

## Dependencies

| Package | Version | Required | Purpose |
|---------|---------|----------|---------|
| Python | ≥ 3.8 | Yes | Language runtime |
| NumPy | ≥ 1.20 | Yes | Numerical arrays, FFT, linear algebra |
| SciPy | ≥ 1.7 | Yes | Interpolation (LinearNDInterpolator), Gaussian filtering |
| Matplotlib | ≥ 3.4 | Yes | All plotting, Mollweide projection |
| h5py | ≥ 3.0 | Yes | HDF5 file I/O (yt bridge) |
| [yt](https://yt-project.org/) | ≥ 4.0 | Yes | Simulation data analysis, projections, slices, FRBs |
| [pyxsim](https://github.com/jzuhone/pyxsim) | — | Yes | CIE spectral models, X-ray emissivity fields (imported unconditionally by `proxy.py`) |
| [astrodendro](https://dendrograms.readthedocs.io/) | — | Optional | Dendrogram analysis (`clump_mass_function.py`) |
| skyproj | — | Optional | All-sky map projections (older `xray.py` notebook) |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KRATOS_VISUAL_DIR` | Path to the kratos `visual/` directory. Required for module imports in notebook and script workflows. | Must be set; typically `~/apps/kratos/visual` |

---

## Physics Overview

### Simulation Setup

The CMZ simulation models the central ~3 kpc of the Milky Way in a Cartesian box with adaptive mesh refinement (AMR) down to ~1 pc resolution. Key elements:

- **Gas dynamics**: Ideal MHD with tabulated CLOUDY cooling/heating, photoionisation heating, and stellar wind/SN feedback.
- **Gravitational potential**: Time-dependent background from stellar bulge, disc, bar, and NFW halo. Bar amplitude ramps from 0 to full strength over 0–146 Myr.
- **Star formation**: Sink particles form above threshold density (n_H > 10⁴ cm⁻³) with Bondi accretion. Each sink particle represents a star cluster.
- **Stellar feedback**: Supernovae (Kimm+ Cen scheme), stellar winds, and ionising radiation from sink particles. Feedback rates from tabulated Starburst99 models.
- **Two runs compared**: b10 (starburst, strong feedback) vs b60 (quiescent, weaker feedback) — differing mainly in feedback efficiency.

### X-ray Emission Model (xray_allsky.py)

The all-sky X-ray map uses pyXsim's CIE (collisional ionisation equilibrium) model:

1. **Volume emissivity**: `CIESourceModel("cloudy")` computes ε(T, n, Z) = n_e n_H Λ(T, Z) [photons cm⁻³ s⁻¹] integrated over 0.6–1 keV.
2. **Per-cell photon rate**: ε × V_cell / (4π) [photons s⁻¹ sr⁻¹] — isotropic emission into 4π sr, 1/(4π) per steradian.
3. **All-sky projection**: Cells are histogrammed by Galactic (l, b) coordinates from the Sun position (8150 pc, 208°).
4. **Pixel value**: Σ(ε·V/(4π)) over all cells along each LOS → total photon rate per steradian [photons s⁻¹ sr⁻¹] in each pixel.

This is the **total photon rate per steradian**, not surface brightness (per area). To convert to eROSITA-like count rates, multiply by telescope effective area × exposure time × pixel solid angle.

### References

- Predehl et al. 2020, Nature 588, 227 — eROSITA X-ray bubbles
- Sormani et al. 2019, MNRAS 488, 4663 — Nuclear disc and bulge models
- Sormani et al. 2020, MNRAS 497, 5024 — NSC and NSD models
- Sormani et al. 2022, MNRAS 512, 1857 — Bar+spiral model
- Portail et al. 2017, MNRAS 465, 1621 — Made-to-measure bar model
- McMillan 2017, MNRAS 465, 76 — Milky Way mass model
