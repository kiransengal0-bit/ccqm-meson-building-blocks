# CCQM Stage-1 Building-Block Calculator

Scientific Python code for **Stage-1 covariant confined quark model (CCQM) building blocks**:

- meson--quark coupling constants `g_H` from the compositeness condition,
- pseudoscalar and vector decay constants `f_P`, `f_V`,
- current-resolved `P -> P` and `P -> V` form factors,
- q² point lists and q² grids,
- readable text/HTML reports, CSV tables, JSON output, plots, and numerical diagnostics.

This package does **not** compute decay widths, branching ratios, CKM-dependent observables, Wilson-coefficient amplitudes, or angular observables. Those are Stage 2 and intentionally outside this repository.

## Quick start

```bash
python -m pip install -r requirements.txt
python ccqm.py run examples/quick_start.txt --output output_quick --precision quick
```

Open:

```text
output_quick/report.txt
output_quick/report.html
output_quick/form_factor_trends.txt
output_quick/form_factor_trends.html
```

## Scientific run

```bash
python ccqm.py run examples/quick_start.txt \
  --output output_high \
  --precision high \
  --convergence strong
```

Precision presets:

```text
quick       n_quad = 4
standard    n_quad = 8
high        n_quad = 12
very_high   n_quad = 16
research    n_quad = 20
```

Convergence modes:

```text
off        no convergence table
basic      compare final n_quad with one lower level
strong     compare three levels
research   compare several levels including n_quad+4
8,12,16    explicit custom quadrature levels
```

## Input-file structure

```text
[global]
lambda_ir = 0.181
Nc = 3
n_quad = 12

[mesons]
# name    kind   M       m1      m2      Lambda    optional_g
B         P      5.279   5.09    0.235   1.88
K         P      0.494   0.424   0.235   1.04

[decay_constants]
B
K

[transition B_to_K]
initial = B
final = K
final_kind = P
m1 = 5.09
m2 = 0.424
m3 = 0.235
currents = scalar, vector, tensor
q2_mode = range
q2_min = 0
q2_max = q2max
q2_points = 20
endpoint = false
```

## Supported currents

For `final_kind = P`:

```text
scalar, pseudoscalar, vector, axial, v_minus_a, v_plus_a, tensor
```

For `final_kind = V`:

```text
scalar, pseudoscalar, vector, axial, v_minus_a, v_plus_a, tensor_plus, tensor_minus
```

## Output folder

```text
output_high/
  results.json
  report.txt
  report.html
  form_factor_trends.txt
  form_factor_trends.html
  tables/
    couplings.csv
    decay_constants.csv
    form_factors.csv
    form_factor_trends.csv
    form_factor_trends_wide.csv
    diagnostics.csv
    convergence.csv
    warnings.csv
  plots/
    *.png
```

Important tables:

- `form_factors.csv`: long table with every q²/current/form-factor value.
- `form_factor_trends_wide.csv`: one row per form factor and one column per q² point, useful for Excel/Origin/Mathematica plotting.
- `diagnostics.csv`: projection condition numbers and endpoint flags.
- `convergence.csv`: numerical stability estimates from repeated quadrature orders.

## Useful commands

Validate input:

```bash
python ccqm.py validate examples/quick_start.txt
```

Create a starter template:

```bash
python ccqm.py template my_input.txt
```

Show supported currents and precision modes:

```bash
python ccqm.py schema
```

## Documentation and validation

- User manual: `docs/user_manual.pdf`
- Example validation against the uploaded Bs CCQM article: `validation/bs_article_2025/validation_report.pdf`

## GitHub upload

```bash
git init
git add .
git commit -m "Initial CCQM Stage-1 building-block calculator"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Choose a license before making the repository public. See `LICENSE_NOT_SELECTED.md`.
