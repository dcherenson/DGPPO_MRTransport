# Distributed NMPC baseline

This implements a planar adaptation of De Carli et al., *Distributed NMPC for
Cooperative Aerial Manipulation of Cable-Suspended Loads*, IEEE RA-L 2025,
[DOI 10.1109/LRA.2025.3604703](https://doi.org/10.1109/LRA.2025.3604703).
The optimization architecture and partition ADMM updates follow the paper;
the robot/load model and attachment constraint follow the requested planar
adaptation. It does not reproduce the 3D Fly-Crane experiments.

The existing spring-coupled plant, rewards, all four safety costs, LiDAR, wind,
policy and training code are preserved. Initial validation shows goal tracking
and obstacle avoidance, but also collisions, nonmonotone consensus residuals
and benchmark QP failures. These are small implementation checks, not estimates
of comparative controller performance.

## Files and dependencies

| File | Purpose |
| --- | --- |
| `dgppo/controllers/__init__.py` | Controller package |
| `dgppo/controllers/dnmpc.py` | Reference, fixed sparse graph, Jacobi ADMM, warm starts and actions |
| `dgppo/controllers/dnmpc_acados.py` | One local ACADOS OCP and the small configuration block |
| `test_dnmpc.py` | Python evaluation loop, original metrics, Rollout, plots and native videos |
| `DNMPC.md` | Formulation, reproduction commands and validation results |
| `README.md` | Link to this note |
| `dgppo/env/vmas_lidar/vmas_collaborative_transport_lidar.py` | Tiny renderer node-count correction: graphs contain agents, LiDAR hits and a padding node; goal/payload are in `env_states` |

ACADOS is installed outside the repository at
`/Users/dmrc/.local/share/acados/v0.6.0`, native ARM64, commit
`503364817c872d474ab5bed219c26760ac267769`. The project's Python 3.10 environment
has the editable `acados_template` interface and CasADi 3.7.2. Activate it with:

```bash
source .venv/acados_env.sh
```

That local, ignored file activates `.venv` and sets `ACADOS_SOURCE_DIR`. Native
libraries use `@loader_path` on this macOS install. A new machine needs its own
[ACADOS installation](https://docs.acados.org/installation/index.html), Python
interface, CasADi, C compiler and template renderer; set `ACADOS_SOURCE_DIR` to
that installation. No ACADOS source or generated C is vendored here. Generated
local libraries are cached under `~/.cache/dgppo_dnmpc`, built at startup and
reused by independent solver capsules, never generated inside the control loop.

## Plant interface and discrepancies

The controller reads exact robot position/velocity and payload pose from
`graph.env_states`. Each active robot receives its own first optimized planar
acceleration directly through `env.step`. Actions have shape `(env.num_agents,2)`;
inactive entries are zero. There is no added inner controller.

The declared `env.dt` is **0.03 s**, so **H=50** and the prediction horizon is
**1.5 s**. The unchanged plant constructs `World` without a `dt` argument and
therefore advances **0.1 s**, with five 0.02 s substeps, per call. The user
explicitly selected prediction with declared `env.dt`; this mismatch is printed
and stored in metadata. Warm starts still shift one prediction stage as requested.

The plant's polygon circumradius is
`R_N = polygon_length/(2*sin(pi/N))`, where `polygon_length=0.2 m`. It is
0.11547/0.14142/0.17013 m for N=3/4/5. The stored `object_length=0.1` is not used
by the plant's attachment geometry. The controller uses the actual plant vertices,
an explicit departure from the initial prompt's assumed `object_length` formula.

The target constructor stores `max_step` but does not forward it to the base
environment, whose `max_episode_steps` remains 256. The new evaluator explicitly
honors `--max-step`; without that option it uses 256, matching `test.py`'s actual
rollout length. The validation runs below use 128 physical steps (12.8 s).
`--obs 0` is rejected because the existing cost function references an undefined
`obs_pos_flat` in that case. The smoke test uses normal random obstacles instead.

NMPC receives noiseless state and ground-truth circular obstacles. The learned
policy uses noisy graph observations and LiDAR; this initial comparison does
not equalize perception. For matched missions, use the same environment parameters,
fixed active count and reset keys, not merely the same seed with different defaults.
The reset-key recipe exactly matches `test.py` and its nested `test_rollout` split:

```python
pool = jax.random.split(jax.random.PRNGKey(seed), 1000)
base = pool[offset + episode]
key_x0, _ = jax.random.split(base, 2)
reset_key, _ = jax.random.split(key_x0, 2)
```

## Local model and objective

Each local OCP contains only one robot and one payload copy:

```text
x_i = [p_ix, p_iy, v_ix, v_iy, p_Lix, p_Liy, theta_Li]   (7 states)
u_i = [a_ix, a_iy, v_Lix, v_Liy, omega_Li]                (5 inputs)
dot(p_i)=v_i; dot(v_i)=a_i; dot(p_Li)=v_Li; dot(theta_Li)=omega_Li
```

The spring forces and full payload dynamics remain in the simulation only.
The local prediction uses multiple shooting with ERK, `SQP_RTI`,
`PARTIAL_CONDENSING_HPIPM` and a Gauss-Newton Hessian. H=50 gives 357 state and
250 control entries per local shooting trajectory; this does not grow with N.
With three obstacles, both modes have 13 stage-cost residuals. Paper mode has
21 stage parameters and 5 path/initial nonlinear constraints; benchmark has
28 parameters and 10 constraints. Terminal constraints number 4 and 9 respectively.
Acceleration component bounds are separate from these nonlinear counts.

At each stage, with wrapped yaw error `e_theta=atan2(sin(theta-ref),cos(theta-ref))`
and `delta_i=p_i-p_Li-R(theta_Li)r_i`, the base cost is exactly:

```text
200 ||p_Li-p_ref||² + 200 e_theta²
 + 8 ||v_Li-v_ref||² + 0.1 (omega_Li-omega_ref)²
 + ||a_i||² + 0.01 ||delta_i||².
```

The terminal cost is `200 ||p_Li-p_ref[H]||² + 200 e_theta[H]²`.
Pose/twist/acceleration weights and the ADMM coefficients are taken from the
paper. The vertex penalty is the requested planar substitute for its cable
configuration penalty. Using the same pose weights at the terminal stage is
the requested implementation choice, not a claimed reproduction of every
terminal weight in the paper.

Costs are summed as discrete stage terms: ACADOS cost scaling is explicitly
one at all H+1 stages, and NLS weights are doubled to cancel ACADOS's `1/2`.
There is no implicit `dt` multiplier or extra `1/2` on the ADMM penalty.

The reference starts at the measured payload pose, advances along a straight
line at at most 1 m/s and rotates along the shortest yaw branch at at most
1 rad/s. Finite differences give consistent stage velocity references. The
position/yaw reference is continuous; stopping at the goal creates a velocity
kink. There is no global path planner. T=1.5 s, K=5, reference speeds, vertex
weight and terminal choice are explicit adaptation settings. No weights or
horizon were tuned during validation.

## Constraints and communication

Paper mode encodes hard constraints at every applicable initial/path/terminal stage:

- `env.action_lim()` component acceleration bounds, currently [-5,5] m/s²;
- `||a_i|| <= 6` m/s²;
- `||p_i-c_obs|| >= agent_radius + obstacle_radius`;
- `||delta_i|| <= env.agent_vertex_constraint`, currently 0.30 m.

There is no invented hard velocity limit. The vertex inequality replaces the
paper's exact Fly-Crane geometry and cable-angle constraints.

Benchmark mode additionally encodes `||p_Li-c_obs|| >= R_N+obstacle_radius`
and `||p_i-p_j_old|| >= 2*agent_radius` for graph neighbors only. The former
uses a conservative payload circle. The latter uses robot trajectories frozen
at the previous ADMM round; their exchange is a benchmark augmentation.
Paper mode exchanges only load-input trajectories. The original environment
still evaluates exact polygon/LiDAR costs and collisions against every active
robot, including non-neighbors. These optimizer constraints are not a safety
certificate for the mismatched plant or a partially converged RTI iterate.

N=3 uses edges (0,1),(1,2). N>=4 uses the fixed cycle with neighbors
`(i-1)%N` and `(i+1)%N`. Degree is at most two and the graph never changes
during a mission. Local decision and constraint dimensions stay constant as N grows.

## Partition ADMM

Let `D=diag(20,20,10)`, `U_i[h]=[v_Lix,v_Liy,omega_Li]` and `d_i=|N_i|`.
At round k, the exact planar version of paper equations (16)-(17) is:

```text
min J_i + sum_h q_i[h]^T U_i[h]
    + sum_h sum_{j in N_i} ||U_i[h] - (U_i_old[h]+U_j_old[h])/2||_D²

q_i_new[h] = q_i_old[h] + D sum_{j in N_i}(U_i_new[h]-U_j_new[h]).
```

For NLS, completing the square gives the equivalent term
`d_i ||U_i-center_i||_D²`, with
`center_i = mean_neighbors((U_i_old+U_j_old)/2) - D^-1 q_i/(2*d_i)`.
Only constants independent of the decision variables are dropped.

All load inputs (and, in benchmark mode only, robot trajectories) are frozen
before any primal solve. Every local solve uses those frozen messages. All
new load inputs are collected before any dual update: this is Jacobi ADMM.
The implementation uses only neighbor means, never an all-agent consensus average.
Five rounds run per physical update; `--admm-iterations` changes that integer.

Initial load inputs equal the reference velocities and q=0; robot guesses use
zero acceleration. At subsequent physical steps, states, controls and duals
shift one stage and repeat their last entry, with measured x[0] imposed again.
Solvers are reset between missions. Every failure prints robot/step/round/status.
The controller reintegrates and checks a previous feasible control sequence;
if no feasible fallback is available it commands zero acceleration. Failed
solver output is never used as a successful solution.

## Validation results

Native ARM64 CPU, seed 1234, episode offset 0, three normal random obstacles,
wind acceleration zero, H=50, K=5. Each row below is **one 128-step mission**.
Safe rate is the original fraction of robots never unsafe in pre-step states;
mission safety also checks the final post-step state. Cost is the original
maximum signed cost, where nonnegative is unsafe. Every row reached all four
existing distance thresholds (0.1/0.2/0.3/0.5 m) at some pre-step state. This is
closest-approach success, not sustained goal regulation; final position errors
were 0.201/0.150/0.048 m in paper mode and 0.292/0.536/0.132 m in benchmark mode.

| Case | Reward | Max cost | Safe rate | Mission safe | Min goal distance m | Failed solves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N=3 paper | -7.4573 | 0.56765 | 0% | 0 | 0.03335 | 0 |
| N=4 paper | -7.4826 | 0.59044 | 0% | 0 | 0.01444 | 0 |
| N=5 paper | -7.4332 | 0.60223 | 0% | 0 | 0.01079 | 0 |
| N=3 benchmark | -15.5799 | 0.65983 | 0% | 0 | 0.02078 | 287 |
| N=4 benchmark | -23.1865 | 1.00000 | 0% | 0 | 0.02041 | 979 |
| N=5 benchmark | -21.2963 | 1.00000 | 0% | 0 | 0.03692 | 627 |

| Case | Mean local solve ms | Mean full update ms | Mean primal residual | Max primal residual |
| --- | ---: | ---: | ---: | ---: |
| N=3 paper | 0.858 | 31.08 | 0.02179 | 0.20162 |
| N=4 paper | 0.851 | 40.88 | 0.02875 | 0.43600 |
| N=5 paper | 0.826 | 49.56 | 0.03943 | 0.48755 |
| N=3 benchmark | 1.333 | 42.84 | 0.26392 | 2.86792 |
| N=4 benchmark | 1.676 | 64.43 | 0.66208 | 3.17244 |
| N=5 benchmark | 1.445 | 74.76 | 0.47810 | 3.29989 |

The separate three-step N=3 smoke test returned reward -1.64368, cost -0.54518,
100% safe rate and mission safety, min distance 1.64402 m, no success thresholds
reached, and no failures. Mean local/full update times were 1.013/31.32 ms.

The obstacle demonstration uses offset 2, N=3, paper mode, 128 steps. A straight
translation of robot 1 toward the payload goal would penetrate obstacle 0's
robot-inflated circle by 0.0803 m. The realized robot 1 path instead deviated
up to 0.2676 m laterally and kept 0.0757 m clearance from that circle. Its
minimum predicted clearance over the saved horizons was 0.0080 m. This run
returned reward -7.93396, max cost 0.65780, 0% safe rate, mission unsafe,
minimum goal distance 0.14601 m, success 0/1/1/1 at the four thresholds and
five failed solves. Mean local/full times were 0.933/33.04 ms. The obstacle
plot demonstrates path modification, not overall mission safety.

First-step primal residuals for rounds 0 through 5 were:

| N | Mode | Residual history |
| --- | --- | --- |
| 3 | paper | 0, .03432, .03757, .02723, .01912, .02216 |
| 4 | paper | 0, .02711, .03822, .03602, .02323, .02407 |
| 5 | paper | 0, .03211, .03989, .04507, .02902, .02653 |
| 3 | benchmark | 0, .67251, .46823, .18796, .51886, .29840 |
| 4 | benchmark | 0, .83309, .54093, .60177, .43688, .50973 |
| 5 | benchmark | 0, .81638, .54099, .80976, .40753, .36785 |

Round 0 is zero because every robot begins with identical reference inputs.
After local constraints create disagreement, later rounds generally reduce
it, but five nonconvex RTI rounds are not monotone and do not prove convergence.
Translational/angular residuals and every local status/time are in the JSON logs.

Benchmark failures were ACADOS status 4 (QP failure), with native HPIPM minimum
step messages. For N=3, the measured initial constraints at sampled failing
steps were all satisfied; these failures cannot simply be called initial-state
infeasibility. The one-stage shift differs from the physical elapsed time and
the nonlinear frozen-neighbor problems can give difficult QP linearizations.
Those are plausible contributors, not isolated causal findings. No solver,
constraint slack, weight or recovery strategy was substituted. Benchmark N=3/4/5
used a zero fallback in 50/114/92 physical updates
respectively (79/260/200 robot updates). Once required, zero is latched for that
entire physical update even if a later ADMM round succeeds. Status zero does not
ensure nonlinear feasibility;
per-robot `predicted_feasible` flags make this limitation visible. Successful
finite RTI iterates supply actions; feasibility checks govern cache eligibility
and diagnostics. For example,
the N=3 paper trajectory had a maximum predicted tether excess of 0.268 m
despite status-zero local solves. Thus enforcing a constraint in the OCP does
not mean that its partially converged iterate or the executed plant satisfies it.
The maximum executed acceleration norms in benchmark N=3/4/5 were
6.000/6.235/6.104 m/s². The final safeguard clips components to the environment's
[-5,5] limits; it does not project the command onto the paper's 6 m/s² ball.
The latter remains a nonlinear OCP constraint, with the same RTI feasibility
limitation. Paper-mode command norms stayed below 2.72 m/s² in these runs.

Full sequential distributed updates take about 31–75 ms on this development
machine, exceeding the declared 30 ms timestep in most runs. Timings exclude
code generation, environment compilation/stepping and plotting. They include
Python parameter updates, trajectory extraction and all N*K local solves.
The horizon remains H=50. The native videos retain the existing 30 fps playback,
so a 128-step video lasts 4.27 s while the plant mission represents 12.8 s.

Main-agent review checked the actual delegated OCP/evaluator source and the
renderer diff. Numeric checks verified the completed-square coefficient, unit
stage scaling and fixed dimensions for N=3/4/5 in both modes. A bounded mock
check verified frozen Jacobi messages, shifted warm starts, padded zeros and
explicit failed-solve zero fallback. Python compilation, CLI and native video
checks passed. Pre-existing edits to `test.py`, `requirements.txt` and the
environment factory were preserved byte for byte. No commit or push was made.

## Reproduction and artifacts

From the repository root, with the activation above:

```bash
python test_dnmpc.py -n 3 --epi 1 --max-step 3 --no-video \
  --output logs/dnmpc_validation/n3_smoke

python test_dnmpc.py -n 3 --epi 1 --seed 1234 --max-step 128 \
  --dnmpc-constraints paper --log --dpi 80 --output logs/dnmpc_validation/n3_paper
python test_dnmpc.py -n 4 --epi 1 --seed 1234 --max-step 128 \
  --dnmpc-constraints paper --log --no-video --output logs/dnmpc_validation/n4_paper
python test_dnmpc.py -n 5 --epi 1 --seed 1234 --max-step 128 \
  --dnmpc-constraints paper --log --no-video --output logs/dnmpc_validation/n5_paper

python test_dnmpc.py -n 3 --epi 1 --seed 1234 --max-step 128 \
  --dnmpc-constraints benchmark --log --dpi 80 --output logs/dnmpc_validation/n3_benchmark
python test_dnmpc.py -n 4 --epi 1 --seed 1234 --max-step 128 \
  --dnmpc-constraints benchmark --log --no-video --output logs/dnmpc_validation/n4_benchmark
python test_dnmpc.py -n 5 --epi 1 --seed 1234 --max-step 128 \
  --dnmpc-constraints benchmark --log --no-video --output logs/dnmpc_validation/n5_benchmark

python test_dnmpc.py -n 3 --epi 1 --seed 1234 --offset 2 --max-step 128 \
  --dnmpc-constraints paper --log --dpi 80 --output logs/dnmpc_validation/n3_obstacle
```

Default wind is zero; `--wind-accel` and `--wind-wavelength` pass through the
existing disturbance unchanged, with no compensation in NMPC. No wind experiment
was run for this initial validation. Remove `--max-step` for 256-step missions.

Each output directory has `statistics.json`, `episode_summary.csv`, `metadata.json`,
per-step `episode_*_diagnostics.json` and compressed trajectory NPZ files. `--log`
adds action/state CSVs, the adapted native-style comprehensive plot, payload
x/y/yaw versus finite-horizon references, first-step ADMM residuals and an XY
obstacle plot. Native `env.render_video` writes MP4s unless `--no-video` is used.
The evaluator stores all local predictions plus original costs and the final state.
All experimental artifacts are under ignored `logs/` directories.
