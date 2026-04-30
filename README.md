# kratos_galaxy

A Python toolkit for setting up and post-processing galaxy-scale hydrodynamic simulations run with the Kratos MHD code. It handles initial condition generation, background potential computation, binary I/O, and data visualization (including yt integration).

---

## Repository Structure

```
kratos_galaxy/
├── base/           # Core data structures and utilities
├── init/           # Initial condition generation
└── visual/         # Post-processing and visualization
```

---

## Modules

### `base/` — Core Utilities

| File | Description |
|---|---|
| `unit.py` | Physical and astrophysical constants in CGS. Defines code unit conversions via user-specified `rho0`, `l0`, `t0`. |
| `data_field.py` | Unit-aware array wrapper. Tracks dimensional units through arithmetic, NumPy ufuncs, and array functions; supports conversion between code and CGS. |
| `grid.py` | N-dimensional grid object supporting Cartesian, cylindrical, and spherical coordinates. Provides meshgrid creation, coordinate transformations, and data interpolation via `LinearNDInterpolator`. |
| `profile_base.py` | Library of standard analytic density profiles (`disc`, `exp`, `gauss`, `uniform`, `sech2`) and their derivatives, used as building blocks for initial condition profiles. |
| `source.py` | Wraps an arbitrary density function as a `source` object with dimensional units. Supports addition and scalar multiplication of source functions. |

---

### `init/` — Initial Condition Generation

| File | Description |
|---|---|
| `main.py` | Entry point. Loads a `.par` configuration file and runs the full IC generation pipeline: `config_setup` → `profile` → `IC`. |
| `init.py` | Defines the `profile` class (density + azimuthal velocity profiles on a cylindrical grid, with optional vertical hydrostatic equilibrium via `zsolve`) and the `IC` class (writes all IC data to a binary file). |
| `background.py` | Defines `background` (compiles all stellar/dark matter source functions from config) and `background_source` (wraps a source with a total mass computed by numerical integration). Includes implementations for multiple published galactic component models. |
| `FDM.py` | Finite-difference Poisson solver (`background_potential`) for numerically computing the gravitational potential on a 2D spherical-polar grid. Used when analytic-only potentials are insufficient. |
| `binary_io.py` | Low-level binary reader/writer used to serialize IC data to disk in Kratos' native format. |

#### Supported Background Source Models

The `background` class in `background.py` includes density functions from the following published models:

- **Sormani et al. 2019** — nuclear disc (`disc_s19`), nuclear bulge (`bulge_s19`), bar (`bar_s19`)
- **Sormani et al. 2020** — nuclear stellar disc (`nsd_s20`), nuclear stellar cluster (`nsc_s20`)
- **Sormani et al. 2022** — stellar disc (`disc_s22`), bar + spiral (`bar_portail` from Portail et al. 2017)
- **McMillan 2017** — thin/thick stellar disc (`disc_m17`)
- NFW dark matter halo and central point source (analytic; used in `phi_analytic` and `dphidR_analytic`)

---

### `visual/` — Post-Processing and Visualization

| File | Description |
|---|---|
| `proxy.py` | Convenience import hub for Jupyter notebooks. Imports all visual tools, enrolls physical constants from the unit system, and sets Matplotlib defaults. |
| `yt_kratos.py` | Bridges Kratos binary outputs to [yt](https://yt-project.org/) by constructing a temporary `.athdf` (Athena++ HDF5) file, enabling yt's full suite of analysis and visualization tools. |
| `hydro_data_gal.py` | Extends the base `hydro_data` reader with galaxy-simulation-specific fields: metallicity, entropy-based pressure, particle data (position, velocity, mass, stellar mass, SFR, SNe count, etc.), mesh stitching, and CIC particle mapping. |
| `plot_gal.py` | High-level plotting helpers: `f_plot_grid` for multi-panel field plots, `time_evo` for time series extraction across simulation outputs, `plt_time_evo` for plotting time series, and derived quantity helpers (`f_par_SFR`, `f_par_Mstar`, `f_par_Mgas`, `bf_hyd_cmzmass`, `bf_hyd_cmzvel`, etc.). |

---

## Configuration File (`.par`)

All major parameters are controlled through a configuration file (e.g., `setup.par`) in INI format. Relevant sections include:

| Section | Controls |
|---|---|
| `[unit]` | Base code units: `rho0`, `l0`, `t0` |
| `[output_grid]` | Grid coordinate system, spacing, extent, resolution |
| `[IC_profile]` | Profile flags per dimension (e.g., `exp`, `disc`, `zsolve`) and output file name |
| `[IC_hydro]` | Gas density normalization (`rhobase`), mean molecular weight (`mu`), initial temperature (`Tdisc`) |
| `[FDM]` | Finite-difference Poisson solver settings: toggle (`FDM_on`), boundary conditions |
| `[background_src]` | Toggle individual source components (e.g., `disc_s19 = 1`) |
| `[background_mass]` | Optionally specify known component masses to skip numerical integration |
| `[background_analytic]` | NFW halo parameters (`rho_halo`, `r_halo`) and black hole mass (`M_bh`) |
| `[zsolve]` | Cutoff radii for the vertical hydrostatic equilibrium solver |

---

## Quickstart

### Generating Initial Conditions

```bash
cd kratos_galaxy/init/
python main.py path/to/setup.par
```

Or programmatically:

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

### Loading Data in a Jupyter Notebook

```python
from visual.proxy import *   # imports units, yt helpers, plot tools

# Load a Kratos output file
d = get_last_dd("output_00010.bin", -1, enroll_prim=True, data_type=galaxy_data)
d.load_particles()

# Quick slice plot
fig = f_plot_grid(d, flds=['rho', 'T_ent', 'met'], axes=[2, 1])
```

### Loading with yt

```python
from visual.yt_kratos import load_kratos_yt
from hydro_data import get_dd

d  = get_dd("output_00010.bin")
ds = load_kratos_yt(d)
slc = ds.slice('z', 0)
slc.plot('density').show()
```

---

## Dependencies

- Python ≥ 3.8
- NumPy
- SciPy
- Matplotlib
- h5py
- yt
- pyxsim (optional; for X-ray synthetic observations)

---

## Environment Variables

| Variable | Description |
|---|---|
| `KRATOS_VISUAL_DIR` | Path to the `visual/` directory. Required for imports in notebook workflows. |
