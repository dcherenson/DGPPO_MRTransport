# Planar distributed NMPC

This controller is a 2D adaptation of De Carli et al., *Distributed NMPC for
Cooperative Aerial Manipulation of Cable-Suspended Loads*, IEEE RA-L 2025,
[DOI 10.1109/LRA.2025.3604703](https://doi.org/10.1109/LRA.2025.3604703).
The local dynamics, cost structure and partition ADMM follow Eqs. (1)–(17).
Geometric coupling uses a one-taut-cable planar analogue at each payload vertex.

There is one formulation, one ACADOS SQP_RTI solver and one matching nominal
plant. The repository supplies obstacle/goal generation and visualization
utilities. Its learned-policy environments and training code remain separate.

## Model and constraints

Robot i has state `x_i=[p_ix,p_iy,v_ix,v_iy,alpha_i]` and each local OCP contains
its own load copy `x_Li=[p_Lx,p_Ly,theta_L]`. The implementation concatenates
these in that order into eight states. Its six inputs are
`[a_ix,a_iy,omega_alpha_i,v_Lx,v_Ly,omega_L]`.

```text
p_dot_i = v_i                 v_dot_i = a_i
alpha_dot_i = omega_alpha_i   p_dot_L = v_L   theta_dot_L = omega_L
phi_i = 2*pi*i/N
e(psi) = [cos(psi),sin(psi)]
b_i = r_b*e(phi_i)             c_i(alpha_i) = e(phi_i+alpha_i)
p_i - p_L - R(theta_L)*(b_i + ell*c_i(alpha_i)) = [0,0]
```

The original paper uses the 3D two-cable Fly-Crane geometry. In a plane,
two fixed-length cables generically constrain a robot to discrete circle
intersections, rather than providing a continuous alpha coordinate. This
implementation therefore uses a paper-compatible one-taut-cable planar
analogue: a fixed attachment on the rigid payload and a rotating cable of
constant length. The paper also identifies single-cable configurations as
applicable to its control methodology.

Attachment vectors are the actual regular-polygon vertices used by the
repository environment and renderer. With its side length 0.2 m,
`r_b=0.2/(2*sin(pi/N))`; for N=3 this is 0.115470054 m. The common configurable
length `ell=0.35 m` is an explicit 2D scene choice. The paper reports 1.1 m
cables for its 3D experimental platform; that physical value is not used for
this planar scene. Cables are drawn from the corresponding rotated payload
vertex to each robot, using these same quantities.

The two geometry components are hard nonlinear equalities at all variable
shooting nodes, including the terminal node. The complete initial state is fixed
and explicitly checked for geometric consistency; redundant initial geometry
rows are omitted from the native QP. Robot circular obstacle clearance is
`||p_i-c_j|| >= robot_radius+obstacle_radius_j`. The reported acceleration norm
limit is 6 m/s², and cable angles lie in `[pi/4,4*pi/6]`. No unreported velocity
limit or component acceleration limit is added.

The dynamics use exact constant-input double/single-integrator updates,
represented by ACADOS ERK multiple shooting. Geometric equality is checked at
shooting nodes and physical update boundaries. This discretization does not
claim exact geometry between nodes: constant angular rates trace a circle while
constant robot acceleration traces a parabola.

With `S=[[0,-1],[1,0]]`, differentiating the geometry gives
`v_i=v_L+omega_L*R*S*(b_i+ell*c_i)+ell*omega_alpha_i*R*S*c_i`.
The implemented stacked `J_2D` has block row
`[I_2, R*S*(b_i+ell*c_i), 0 ... ell*R*S*c_i ... 0]` and maps
`[v_Lx,v_Ly,omega_L,omega_alpha_0,...]` to stacked robot velocities.

For N=3, eliminating the three cable angular rates leaves the matrix with rows
`A_i=[cos(phi_i+alpha_i),sin(phi_i+alpha_i),r_b*sin(alpha_i)]`.
For symmetric phases its determinant is
`sqrt(3)*r_b/2*(4*prod(sin(alpha_i))+sin(sum(alpha_i)))`.
On `[pi/4,4*pi/6]^3`, `4*prod(sin(alpha_i))>=sqrt(2)`, so
`det(A)>=sqrt(3)*r_b/2*(sqrt(2)-1)>0`.
Since cable length is positive, this establishes full column rank six for the
planar Jacobian throughout the selected N=3 domain. The validation checks this
formula, numerical rank and analytic derivatives against finite differences
before permitting any native primal solve. Changing load yaw multiplies the
Jacobian on the left and right by orthogonal rotations, so its singular values
are independent of yaw; this property is also tested.

## Objective and paper mapping

The load stage cost is
`200||p_L-p_L^d||² + 200*wrapped_yaw_error² + 8||v_L-v_L^d||²
+ 0.1*(omega_L-omega_L^d)²`.
The robot stage cost is
`0.01*(alpha_i-alpha_i^d)² + 0.01*(omega_alpha_i-omega_alpha_i^d)²
+ ||u_i-u_i^d||²`, with `u_i=[a_ix,a_iy,omega_alpha_i]`.
Thus the angular-rate input appears in both its cable-rate term and the full
robot-input term, as required by the requested cost structure.

Physical stage terms use Euler quadrature with a dt factor. The ADMM trajectory
penalty remains unscaled by dt. NLS weights include a factor of two to cancel
ACADOS's one-half convention. Terminal load pose and alpha costs follow paper
Eq. (10), using explicitly chosen numeric weights equal to their stage weights.

| Paper quantity | 2D implementation | Exact/adapted/unspecified |
| --- | --- | --- |
| Robot double integrator, Eq. (1) | Two position/velocity components | Adapted dimension; same dynamics |
| Load kinematics, Eq. (2) | Two translations and scalar yaw | Adapted from quaternion/3D rotation |
| Cable angle integrator, Eq. (3) | One alpha and omega_alpha per robot | Exact structure |
| 3D two-cable Fly-Crane geometry, Eq. (4) | One taut planar cable from each payload vertex | Adapted geometry |
| Payload attachment vectors | Existing polygon vertices, side 0.2 m | Repository scene geometry |
| Cable length (3D platform: 1.1 m) | Common configurable length 0.35 m | Explicit 2D implementation choice |
| Velocity Jacobian, Eq. (6) | 6x6 J_2D, full column rank on selected N=3 alpha domain | Adapted dimension; verified rank |
| Cable-angle bounds | pi/4 to 4*pi/6 | Exact reported values |
| Acceleration norm | 6 m/s² | Exact reported value |
| Load weights | Position/yaw 200, velocity 8, angular rate 0.1 | Reported values with adapted dimensions |
| Cable/input costs | Alpha/rate 0.01; identity robot-input weight | Reported structure/values with adapted dimensions |
| ADMM penalties, Eqs. (16)–(17) | diag(20,20,10), complete graph | Exact reported values and N=3 graph |
| Terminal numeric weights | Pose 200, alpha 0.01 | Unspecified numerically; explicit implementation choice |
| T, H, prediction/update interval | 1.5 s, 45 stages, 1/30 s | Retained physical horizon; nominal update interval matches the paper’s 30 Hz reference transmission |
| Desired alpha and cable rate | Midpoint of bounds; rate zero | Unspecified numerically; explicit implementation choice |
| Desired robot acceleration | Zero | Explicit implementation choice |
| Desired load trajectory | Quintic time-polynomial translation and shortest yaw; peak speeds <=1 m/s and <=1 rad/s | Time-polynomial path follows the reported operating style; degree, duration and yaw interpolation are explicit 2D choices |
| Robot/obstacle size | Repository circles; robot radius 0.09 m | Adapted mission geometry |
| Local primal solver | ACADOS SQP_RTI, one call per ADMM primal | Exact reported RTI architecture |
| Hessian/QP/integrator | Gauss-Newton, partial-condensing HPIPM, ERK | Requested implementation choices |

## Partition ADMM and execution

Every robot communicates with every other robot. Only local load-input
trajectories `U_i[h]=[v_Lx,v_Ly,omega_L]` are exchanged. With
`D=diag(20,20,10)` and degree `d_i=N-1`, each Jacobi round freezes all previous
messages before solving any local primal:

```text
min J_i + sum_h q_i[h]^T U_i[h]
    + sum_h sum_(j in neighbors_i) ||U_i[h]-(U_i_old[h]+U_j_old[h])/2||_D²
q_i_new[h] = q_i_old[h] + D*sum_(j in neighbors_i)(U_i_new[h]-U_j_new[h])
```

Completing the square gives the native NLS center
`mean_neighbors((U_i_old+U_j_old)/2)-q_i/(2*d_i*diag(D))` and coefficient
`d_i*D`. Each primal makes exactly one standard SQP_RTI preparation/feedback
call. All new inputs are collected before dual updates. The conditional
physical-update test is configured for five rounds.

A mission starts with robots reconstructed from the sampled load pose, bounded
nominal alpha, polygon attachment radius and cable length. The reset checks
obstacle clearance. The initial horizon follows the smooth reference described
below and is checked for geometry, exact integration and all bounds before any
primal solve. Subsequent warm starts shift the previous local trajectories and
ADMM duals one stage, repeat the final entry and impose the new measured initial
state. Solver capsules reset between missions.

The nominal plant directly integrates these same robot, load and alpha
kinematics in float64. A physical update is committed only when every final
local trajectory is explicitly feasible and the physical candidate checks pass.
Robot zero's first load input supplies the candidate load motion; the other
load-input copies remain local proposals. Before committing, the plant checks the integrated candidate's
geometry, cable angles, acceleration and obstacle clearance. Inconsistent
updates are rejected atomically and diagnosed, as selected by the user. There
is no control averaging, geometric projection, action clipping or fallback.

Finite ADMM rounds do not guarantee consensus. First-input and full-horizon
translational/angular disagreement are recorded as diagnostics, and no longer
gate execution. Integrated geometry and the other physical checks can still
reject a proposal even when each local OCP is feasible relative to its own load copy.
The external execution/diagnostic geometry acceptance tolerance is 1e-3 m
(1 mm). This is an implementation acceptance tolerance, not a parameter from
De Carli et al. It does not soften the native OCP: both ACADOS geometry
components still have exactly zero lower and upper bounds, and native NLP/QP
solver tolerances are unchanged. Acceleration, alpha, obstacle and exact-dynamics
checks retain their prior 1e-6 thresholds. The earlier first-input limits of
1e-3 m/s and 1e-3 rad/s remain in diagnostic metadata for historical comparison,
with `consensus_gates_execution=false`; neither is an execution limit.
The combined first-input norm and full-horizon ADMM residual also remain
diagnostics. NLP residuals are recorded separately in native units.

Acceptance requires outer RTI status 0, explicit trajectory feasibility and
the plant consistency checks. A QP maximum-iteration return is recorded but
does not itself reject an otherwise feasible proposal, following the native
RTI status policy. Acceptance therefore establishes physical feasibility, not
converged NLP optimality. Nonfinite states or any nonfinite control channel
terminate the ADMM attempt after recording the current round.

## Files and reproduction

| File | Purpose |
| --- | --- |
| `dgppo/controllers/dnmpc_acados.py` | The single eight-state/six-input OCP and configuration |
| `dgppo/controllers/dnmpc.py` | Reference, frozen-message partition ADMM, RTI and feasibility diagnostics |
| `dgppo/env/planar_transport.py` | Mission-generation adapter, nominal plant and visualization |
| `dgppo/env/planar_geometry.py` | Attachment/cable positions, analytic Jacobian and domain rank checks |
| `dgppo/env/planar_safety.py` | True robot-obstacle clearance over each executed ZOH interval |
| `test_dnmpc.py` | Short nominal evaluation, logs, plots and animation |
| `run_dnmpc_bias_experiment.py` | Nominal validation followed conditionally by 20 paired missions per bias level |
| `validate_dnmpc_geometry.py` | Geometry audit and gated single-local/one-physical-update validation |
| `sweep_dnmpc_frozen_admm.py` | Reconstruct the saved rejected step and compare frozen ADMM budgets without plant execution |

Local ACADOS v0.6.0 is installed at
`/Users/dmrc/.local/share/acados/v0.6.0`. The repository's Python environment
contains `acados_template` and CasADi. On this machine:

```bash
source .venv/acados_env.sh
python -m unittest discover -s tests -p 'test_planar*.py' -q
python test_dnmpc.py -n 3 --epi 1 --seed 1234 --offset 0 --obs 3 \
  --max-step 300 --obstacle-bias 0 --log --no-video \
  --output logs/dnmpc_perception_shift/nominal_seed1234
```

This command attempts a 10 s mission (300 physical updates), each with 20 ADMM rounds
and one SQP_RTI call per local primal. There is no preliminary local solve.
A plant step commits only when the final local trajectories and the physical
candidate checks pass; otherwise the mission stops after recording the
unexecuted candidate. The nominal plant supplies zero wind and the graph is
complete. Disturbance sweeps are conditional on a successful nominal validation. Cable
length remains configurable through `DNMPCConfig` and `--cable-length`.
The separate `validate_dnmpc_geometry.py` utility retains its geometry-audit,
single-local and conditional physical-update stages.

Generated native libraries are cached outside the repository. On another
machine follow the [ACADOS installation instructions](https://docs.acados.org/installation/index.html)
and configure its Python interface and `ACADOS_SOURCE_DIR`.

The nominal model has no external wind force. Its mission seed uses the same
nested JAX reset-key recipe as the repository evaluator. Output metadata makes
all implementation choices explicit. Diagnostics retain each primal's native
status/residuals, geometry error at every stage, bounds, clearance, consensus
residual history and timing. A failed update terminates the test and keeps the
unexecuted candidate separate from actual state snapshots.

## Smooth reference and horizon initialization

The DNMPC and nominal plant use `dt=1/30 s`; the unchanged 1.5 s physical
prediction horizon now has H=45 intervals and 46 nodes. The paper reports
sending reference position, velocity and acceleration to each low-level
quadrotor controller at 30 Hz. This implementation adopts that interval for
its nominal model and controller update. The paper does not specify this
1.5 s horizon or the discretization used here.

The paper describes a straight path parameterized by a time polynomial.
The implemented reference uses the quintic blend

```text
tau = min(t/T_ref, 1)
s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
z_ref(t) = z_current + s(tau)*Delta_z
Delta_z = [goal_px-current_px, goal_py-current_py, shortest_yaw_difference]
T_ref = max(1.5, (15/8)*||Delta_p||/v_ref_max,
                   (15/8)*|Delta_theta|/omega_ref_max)
```

Its velocity and acceleration are analytic derivatives. Both are zero at the
start and endpoint; the reference holds the goal after T_ref. The factor 15/8
is the blend's peak normalized rate and preserves the existing reference
speed caps, 1 m/s and 1 rad/s. These are reference-generation limits, not new
OCP velocity constraints. A configured zero cap holds that channel. For the
seed-1234 mission, T_ref=3.085224836 s, so the prediction horizon samples the
first 1.5 s of this motion. The polynomial degree, duration formula and smooth
yaw interpolation are explicit implementation choices rather than claimed
paper parameters. At each update the reference is generated from the current
load pose toward the goal.

For initialization, each alpha stays at its measured initial value. At every
shooting node, robot positions are constructed using the unchanged exact
one-taut-cable geometry and the sampled reference load pose. Controls and
robot velocities then satisfy the same constant-input discrete dynamics:

```text
u_L[h] = (z_ref[h+1]-z_ref[h])/dt
omega_alpha_i[h] = 0
a_i[h] = 2*(p_i[h+1]-p_i[h]-dt*v_i[h])/dt^2
v_i[h+1] = v_i[h] + dt*a_i[h]
v_i[0] = measured_robot_velocity_i
```

The load inputs are interval-average pose rates; desired velocities in the
cost remain the analytic polynomial derivatives. Likewise, discrete robot
accelerations need not equal the continuous derivative at a node. Using
analytic robot velocities directly would generally break the discrete
position equations. This construction enforces both exact geometry at nodes
and exact model integration, without projecting a proposed plant update.
The initial reference velocity and acceleration are zero, and the measured
initial robot velocities are zero. All local load-input horizons agree, so
the unchanged ADMM center for the first solve equals their common horizon;
initial duals remain zero. Initialization is explicitly rejected if any
constraint fails rather than modifying the reference or bounds.

## Earlier 30 Hz single-local diagnostic

The following saved diagnostic used the previous external geometry tolerance
of 1e-6 m. Its native result remains valid, but its acceptance outcome describes
that earlier threshold. The current 1 mm execution attempt is reported below.

The test uses N=3, seed 1234, offset 0, three obstacles, zero wind and the
complete graph. All 19 tests pass, covering polynomial endpoint conditions,
analytic derivatives, speed caps, exact warm-start geometry/dynamics, and
model/execution consistency. A code/configuration comparison confirms that
only the interval, reference, initial warm start and diagnostic scope changed:
geometry, cost coefficients, ADMM equations/penalties, bounds and solver
settings remain unchanged. Stage quadrature automatically uses the new dt;
physical cost coefficients and the unscaled ADMM penalty retain their values.

The unchanged geometry audit covers 31³ grid points plus 20,000 random samples:

| Geometry audit quantity | Result |
| --- | ---: |
| Minimum sampled J_2D rank | 6 |
| Minimum sampled J_2D singular value | 0.0422359250 |
| Minimum sampled reduced-matrix rank | 3 |
| Minimum evaluated analytic/numerical determinant | 0.1155394517 |
| Proven determinant lower bound over the whole domain | 0.0414213562 |
| Maximum determinant formula disagreement | 2.22e-16 |
| Maximum Jacobian/finite-difference disagreement | 1.41e-10 |

All three initial local horizons are feasible. Their maximum geometry error is
5.78e-16 m, maximum acceleration norm is 1.251616225 m/s², minimum robot-obstacle
clearance is 0.019384764 m and exact-integration discrepancy is zero. Alpha stays
at 1.439896633 rad. The reference's initial velocity and acceleration are
exactly zero. Robot 0's warm-start geometry error is 2.62e-16 m.

Only robot 0 makes one SQP_RTI call, with the reference and ADMM messages fixed:

| First local SQP_RTI result | Value |
| --- | ---: |
| Native calls / explicit feasible trajectories | 1 / 0 |
| Warm-start geometry error, robot 0 | 2.62e-16 m |
| Outer RTI status | 0 |
| Actual QP status / iterations | 0 / 5 |
| Maximum nonlinear geometry error after RTI | 0.000666255378 m, stage 23 at 0.766667 s |
| Geometry acceptance tolerance | 1e-6 m |
| Maximum acceleration norm / violation | 0.574158416 m/s² / 0 |
| Alpha range / violation | [1.417930579, 1.501448471] rad / 0 |
| Minimum robot-0 obstacle clearance / violation | 0.623679001 m / 0 |
| Maximum exact-integration discrepancy | 4.00e-15 |
| NLP stationarity residual | 0.142246104 |
| NLP equality residual | 8.88e-16 |
| NLP inequality residual | 0.000620192440 |
| NLP complementarity residual | 0.00680126601 |
| Native solve time | 2.371458 ms |
| Solve/retrieval wall time | 2.579292 ms |
| ADMM residual | Not measured: ADMM was not run |
| Physical update committed | No; no update attempted |

The four NLP residuals are recomputed after the call. ACADOS represents the
geometry equality as nonlinear `h` rows with equal lower/upper bounds, so its
error appears in the native inequality residual. The small native equality
residual reflects the shooting dynamics. Native setup and parameter/warm-start
loading are excluded from solve timings.

An independent inspection of the saved iterate reproduces all constraint
violations. The geometry linearized at the reference-based warm start is
satisfied to a maximum norm of 3.72e-16 m. At the worst node, the RTI update
changes alpha by 0.061551838 rad and yaw by 0.000153716 rad, changing the cable
phase by 0.061705554 rad. The resulting nonlinear Taylor remainder has norm
0.000666255378 m and accounts for the geometry error to roundoff. The QP solves
its tangent constraints, but this finite angular correction does not preserve
the original trigonometric equality within tolerance. The first future node's
geometry error is only 5.35e-8 m; acceptance nevertheless requires feasibility
of the complete local trajectory.

The first-local gate therefore stops this case. No other robot is solved, no
five-round ADMM attempt is made and no physical update is committed. The
reference, geometry, weights, bounds, tolerance and RTI call count are not tuned
in response to the result. Actual goal progress is zero.

Artifacts are saved under `logs/dnmpc_30hz_smooth/n3_seed1234/`:

- `geometry_validation.json`: sampled rank/determinant results.
- `diagnostic.json`: native result, reference conditions, warm-start feasibility, frozen-message witness and gate outcome.
- `local_predictions.npz`: mission, analytic reference velocity/acceleration, initial states/controls, frozen messages and the returned iterate.
- `offline_analysis.json` and `geometry_linearization.npz`: independent feasibility and Taylor-remainder checks.
- `first_local_diagnostic.png`: warm-start and post-RTI constraint diagnostics.

## Earlier one-update diagnostic with the strict consensus gate

This historical test used the old combined first-input norm threshold of 1e-6. Its rejection is superseded by the separate consensus-gate test below; the geometry tolerance was already 1e-3 m.

N=3, complete graph, three obstacles, zero wind, seed 1234, offset 0. One physical control attempt used five ADMM rounds and exactly one SQP_RTI call per local primal: 15 calls total. No preliminary local solve or subsequent update was run.

All violation columns below are maxima over the three local trajectories. Status and QP-iteration triplets are ordered by robot 0, 1, 2. QP status 2 is a maximum-iteration return; robot 2 reached the unchanged 50-iteration limit in every round.

| Round | Max geometry (m) | Accel violation (m/s²) | Alpha violation (rad) | Obstacle violation (m) | RTI statuses | QP statuses | Feasible locals |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 0.0265241816 | 0 | 0 | 0 | 0,0,0 | 0,0,2 | 1/3 |
| 2 | 0.00415248914 | 0 | 0 | 0 | 0,0,0 | 0,0,2 | 2/3 |
| 3 | 0.00111686261 | 0 | 0 | 0 | 0,0,0 | 0,0,2 | 2/3 |
| 4 | 0.000235946788 | 0 | 0 | 0 | 0,0,0 | 0,0,2 | 3/3 |
| 5 | 5.99247959e-05 | 0 | 0 | 0 | 0,0,0 | 0,0,2 | 3/3 |

NLP residuals are recomputed after each solve. Values below are componentwise maxima over the three robots. ADMM residuals are maximum pairwise differences across the entire 45-interval load-input horizon.

| Round | NLP stat | NLP eq | NLP ineq | NLP comp | Translation residual (m/s) | Angular residual (rad/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.477848956 | 1.33226763e-15 | 0.0255834852 | 2.54953595 | 0.00797594636 | 0.00129575874 |
| 2 | 0.534916731 | 1.33226763e-15 | 0.00369101053 | 1.8361532 | 0.00490860121 | 0.000463885622 |
| 3 | 0.303534442 | 8.8817842e-16 | 0.00102877289 | 0.117887086 | 0.00261357522 | 0.000323545627 |
| 4 | 0.169254494 | 8.8817842e-16 | 0.000213387395 | 0.022491696 | 0.00197583774 | 0.000402591391 |
| 5 | 0.0892138723 | 1.33226763e-15 | 5.465446e-05 | 0.00587133728 | 0.00119450594 | 0.000376318503 |

| Round | QP iterations | Sum native solve time (ms) | Sum solve/retrieval wall time (ms) |
| --- | --- | ---: | ---: |
| 1 | 5,4,50 | 6.524957 | 7.477375 |
| 2 | 5,4,50 | 4.876874 | 5.582540 |
| 3 | 5,4,50 | 5.032874 | 5.818625 |
| 4 | 5,4,50 | 4.697748 | 5.464123 |
| 5 | 5,5,50 | 4.380749 | 5.024208 |

Total native solve time was 25.513202 ms; total solve/retrieval wall time was 29.366871 ms. The full controller update, including warm-start construction, parameter loading, diagnostics and ADMM operations, took 63.196584 ms. Per-round timings are sums of the three recorded solve calls, not complete round wall times. Native setup/code generation and the plant preview are excluded. This measured serial controller update exceeded the 33.333 ms nominal interval; no repeated timing benchmark was run.

The final local proposal (round 5) satisfies geometry <=1e-3 m and all other explicit checks:

| Robot | Max geometry (m) | Max acceleration (m/s²) | Alpha range (rad) | Min obstacle clearance (m) | RTI/QP | NLP [stat,eq,ineq,comp] |
| --- | ---: | ---: | --- | ---: | --- | --- |
| 0 | 4.65222388e-07 | 0.578625695 | [1.43648186, 1.50529927] | 0.623763705 | 0/0 | 0.0101281708, 8.8817842e-16, 3.67914974e-07, 8.69330444e-06 |
| 1 | 1.04648964e-06 | 0.93117007 | [1.33253914, 1.47811571] | 0.226662201 | 0/0 | 0.010150767, 8.8817842e-16, 9.83155042e-07, 1.89019624e-05 |
| 2 | 5.99247959e-05 | 1.12267506 | [1.43947262, 1.7372992] | 0.0467479064 | 0/2 | 0.0892138723, 1.33226763e-15, 5.465446e-05, 0.00587133728 |

Final acceleration, alpha and obstacle violations are zero for every robot. The maximum final exact-integration discrepancy is 4.00e-15. Outer RTI status 0 and explicit local feasibility pass under the existing acceptance policy; the QP maximum-iteration return is recorded separately and does not establish NLP optimality.

The unexecuted plant candidate has:

| Candidate check | Value | Result |
| --- | ---: | --- |
| Maximum nonlinear geometry error | 8.74634548e-06 m | Pass: <=1e-3 m |
| Maximum acceleration / violation | 0.731266864 m/s² / 0 | Pass |
| Alpha range / violation | [1.43899806, 1.44030079] rad / 0 | Pass |
| Minimum obstacle clearance / violation | 0.173901843 m / 0 | Pass |
| First-input translational disagreement | 0.000230803556 m/s | Consensus fails |
| First-input angular disagreement | 0.000103849749 rad/s | Consensus fails |
| Existing combined first-input disagreement | 0.000251985048 | Fail: >1e-6 |
| Committed physical steps | 0 | Rejected |

The exact rejection reason is `shared_load_disagreement`. The largest combined first-input difference is between robots 0 and 2: 0.000251985048, about 251.985 times the unchanged consensus threshold. The largest angular difference is between robots 1 and 2. The preview uses robot 0’s load input only to show the unexecuted candidate; it neither averages controls nor commits that candidate. Five finite ADMM rounds reduced the horizon residuals but did not yield a consistent shared load input for execution. No second attempt, extra RTI pass, changed penalty, tolerance or solver setting followed this rejection. Actual goal progress is zero.

All 20 tests pass, including separate geometry/non-geometry and consensus threshold checks. Code/configuration comparison confirms unchanged native solver/OCP functions, geometry, reference, timestep, cost coefficients and ADMM equations. Independent inspection reproduces all 15 geometry profiles, exact dynamics, bounds, horizon residuals and first-input disagreement without any additional solver calls.

Artifacts are saved under `logs/dnmpc_1mm_admm5/n3_seed1234/`: `round_report.json` contains all round and final-candidate data; `episode_diag.json` retains every local status/residual/timing; `episode_0000.npz` contains all round trajectories and the unchanged physical state; `statistics.json` and `metadata.json` record the outcome and configuration.

## Earlier nominal mission with separate first-input consensus thresholds

Only the execution consensus gate changed: velocity agreement <=1e-3 m/s and angular-rate agreement <=1e-3 rad/s, each measured as the maximum pairwise difference of the first load-input copies. The combined norm remains recorded. Full-horizon ADMM residuals and all OCP/configuration settings are unchanged, including the 1 mm geometry acceptance tolerance, 1/30 s timestep, 45-interval horizon, smooth reference, five ADMM rounds and one SQP_RTI call per local primal.

The seed-1234, offset-0, N=3 nominal mission used a complete graph, three obstacles and zero wind. It requested 20 physical steps and stopped at the first rejection: **one committed update and one rejected update**, at zero-based physical step 1. Exactly 30 local RTI calls were made; there were no additional solves after rejection. Only 1/30 s of physical time was committed.

| Physical step | Outcome | Final local geometry max (m) | Plant candidate geometry max (m) | First velocity error (m/s) | First angular error (rad/s) | Controller update (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Committed | 5.99247959e-05 | 8.74634548e-06 | 0.000230803556 | 0.000103849749 | 52.602458 |
| 1 | Rejected | 8.74634548e-06 | 0.000105408546 | 0.00315705266 | 0.000717819507 | 45.363667 |

The exact failing condition is `shared_load_velocity_disagreement`: robots 0 and 1 differ by **0.00315705265956 m/s > 0.001 m/s**. Their excess is 0.00215705265956 m/s. All three robot pairs exceed the velocity threshold at this step: 0–1 = 0.00315705266, 0–2 = 0.00155554599, 1–2 = 0.00167793580 m/s. The maximum angular difference is between robots 1 and 2 and passes. All final local trajectories pass explicit feasibility, and the integrated candidate passes geometry, acceleration, alpha and obstacle checks. The rejected candidate was not committed, averaged or projected. Five finite ADMM rounds did not bring the second update's first velocity inputs within the execution limit.

Local feasibility is **26/30 = 86.67%** over all rounds and **6/6 = 100%** over the final-round trajectories. Four intermediate local trajectories at step 0 failed geometry; none were committed. Maximum nonlinear geometry error over all 30 trajectories was 0.0265241816 m (0.0255241816 m above the implementation acceptance limit). Over final-round trajectories it was 5.99247959e-05 m, with zero excess above 1 mm. The maximum geometry error in actual committed states was 8.74634548e-06 m. The rejected plant candidate's error was 0.000105408546 m. Acceleration, alpha and obstacle violations were **zero** over all local trajectories and both plant candidates. Maximum local discrete-dynamics discrepancy was 2.10853557e-12.

Full-horizon residuals below retain their original definition: maximum pairwise differences across all 45 load-input intervals. They are diagnostics and do not gate execution. In particular, step 0 committed despite its full-horizon velocity residual exceeding 1e-3 m/s.

| Step | ADMM round | Feasible locals | Max geometry (m) | Full-horizon combined residual | Full-horizon velocity (m/s) | Full-horizon angular (rad/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1/3 | 0.0265241816 | 0.00803370665 | 0.00797594636 | 0.00129575874 |
| 0 | 2 | 2/3 | 0.00415248914 | 0.00490998598 | 0.00490860121 | 0.000463885622 |
| 0 | 3 | 2/3 | 0.00111686261 | 0.00261501364 | 0.00261357522 | 0.000323545627 |
| 0 | 4 | 3/3 | 0.000235946788 | 0.00197804204 | 0.00197583774 | 0.000402591391 |
| 0 | 5 | 3/3 | 5.99247959e-05 | 0.00119453912 | 0.00119450594 | 0.000376318503 |
| 1 | 1 | 3/3 | 0.000409565215 | 0.00737290164 | 0.00737287706 | 0.00122388061 |
| 1 | 2 | 3/3 | 0.000349856640 | 0.00499018573 | 0.00494473502 | 0.00134534962 |
| 1 | 3 | 3/3 | 1.05549709e-05 | 0.00298164462 | 0.00292299468 | 0.000658935519 |
| 1 | 4 | 3/3 | 8.74634548e-06 | 0.00199731654 | 0.00199711114 | 0.000199247566 |
| 1 | 5 | 3/3 | 8.74634548e-06 | 0.00315711164 | 0.00315705266 | 0.000717819507 |

There were **0/30 outer RTI failures**. QP status was 0 for 20 calls and 2 (maximum iterations) for 10 calls; robot 2 reached the unchanged 50-iteration QP limit in every round of both attempts. Every round's RTI status triplet was 0,0,0 and QP triplet was 0,0,2. These QP returns remain recorded under the existing feasibility-based acceptance policy.

Final-round NLP residual componentwise maxima over the three robots were:

| Step | Stationarity | Equality | Inequality | Complementarity |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.0892138723 | 1.33226763e-15 | 5.465446e-05 | 0.00587133728 |
| 1 | 0.00416938546 | 1.33226763e-15 | 2.15693704e-07 | 2.24788530e-05 |

Controller update time averaged 48.983063 ms (range 45.363667–52.602458 ms), exceeding the 33.333333 ms update interval in both attempts. Native solve time summed to 24.752328 ms at step 0 and 19.615827 ms at step 1; solve/retrieval wall time summed to 27.549207 and 22.298288 ms respectively. Per-call native time averaged 1.478939 ms and solve/retrieval time averaged 1.661583 ms. These are the measured serial calls from this short test, excluding native setup, plant preview, plotting and logging from controller update time.

Actual payload goal distance decreased from 1.64545324612 to 1.64536089043 m: **0.000092355692 m (0.092356 mm) progress**. The final committed load pose was [1.26237176593, 1.88187753397, 3.01903989987]. Goal progress excludes the rejected candidate.

All 21 model/geometry/reference tests pass, including inclusive consensus limits and independent velocity/angular acceptance when the combined norm exceeds 1e-3. Personal diff and AST review confirms unchanged OCP functions/configuration, controller/ADMM implementation, geometry, reference, timestep and plant integration. Independent analysis of saved arrays reproduced all 30 geometry profiles, dynamics and bounds, all 10 horizon residuals, first-input pairwise errors, both candidate integrations and the single committed state without further solver calls. The test stopped at rejection; no tuning, ADMM sweep or additional mission followed.

Artifacts are in `logs/dnmpc_consensus_1mm_20step/n3_seed1234/`. `consensus_report.json` contains the independently verified per-round and per-attempt summary; `episode_diag.json` preserves every local status, NLP residual and timing; `episode_0000.npz` retains trajectories and actual committed states; `statistics.json` and `metadata.json` contain aggregate results and the unchanged controller configuration.

## Earlier frozen rejected-step ADMM budget sweep

The corrected planar formulation and default controller configuration remain unchanged. This diagnostic varies only the local control update's ADMM budget, using K=5,10,20,40 at the rejected zero-based physical step 1 of the seed-1234, offset-0, N=3, complete-graph mission with three obstacles and zero wind. **No plant update was executed and the 20-step mission was not resumed.** The first-input velocity/angular acceptance limits remain 1e-3 m/s and 1e-3 rad/s; geometry acceptance remains 1e-3 m.

The saved trajectory archive did not contain ACADOS multipliers or ADMM duals, so the preceding step-0 control update was replayed without plant execution to reconstruct them. Its complete state/control round arrays matched the archive exactly (maximum difference zero). The snapshot contains the previous trajectories, duals, reference, step index, all native iterate fields (x,u,z,sl,su,pi,lam), native stage parameters, and saved step-1 measured state, goal and obstacle data. Each budget restores this same snapshot before calling the unchanged controller and its existing warm-start shift once. The reference was bitwise identical to the saved step-1 reference in every case. Native QP warm start remains disabled; solver reset/iterate restoration changes no solver settings. ADMM messages and duals then evolve within each update using the existing frozen-message Jacobi formulation.

K=5 exactly reproduced the original rejected step-1 round arrays. All shorter budget runs were independently verified to be bitwise identical prefixes of K=40. This establishes that the comparison uses the same initial controller data rather than continuing one budget's final state into the next. The plant step method is blocked in the diagnostic script.

**K=10 is the smallest successful budget among the four requested values.** Selection requires all final local trajectories to be explicitly feasible, outer RTI status 0, and both first-input consensus limits to pass. It does not require full-horizon consensus below these thresholds. No default budget, weights, rho, solver settings, geometry, reference, timestep, tolerances or warm-start logic were changed.

| K | First velocity (m/s) | First angular (rad/s) | Horizon velocity (m/s) | Horizon angular (rad/s) | Final geometry max (m) | Feasible robots | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 5 | 0.00315705266 | 0.0007178195069 | 0.00315705266 | 0.0007178195069 | 8.746345482e-06 | 0,1,2 all feasible | Fail: velocity consensus |
| 10 | 0.0003155473811 | 4.919709223e-05 | 0.000694908793 | 0.0002125631315 | 8.746345482e-06 | 0,1,2 all feasible | Pass |
| 20 | 0.001022949935 | 2.021304156e-06 | 0.001022949935 | 2.857718878e-05 | 8.746345482e-06 | 0,1,2 all feasible | Fail: velocity consensus |
| 40 | 0.0004126865546 | 3.129214983e-06 | 0.0004126865546 | 6.226335594e-06 | 8.746345482e-06 | 0,1,2 all feasible | Pass |

At K=20 the velocity error exceeds its limit by 2.29499351e-05 m/s, despite smaller angular error and feasible local trajectories. Increasing K is therefore not a monotonic acceptance improvement. The velocity residual exhibits a decreasing trend with oscillations: full-horizon velocity drops from 0.00737287706 at round 1 to 0.000412686555 at round 40 (94.40% reduction), but rises during rounds 4–6, 13–21 and 37–40. It passes the first-input limit at round 10, rises above it again at rounds 20–23, then passes again from round 24 onward through round 40. The angular horizon residual falls from 0.00122388061 to 6.22633559e-06 (99.49% reduction), with smaller intermediate rises. This finite test shows decreasing residuals with oscillation, not a monotonic convergence guarantee or proof of a limiting residual floor. No rho tuning or threshold relaxation followed.

All final trajectories satisfy geometry acceptance. Their maximum error is 8.74634548e-06 m for every budget, including the fixed measured initial node; excess over 1 mm is zero. Geometry over all rounds/robots peaks at 0.000409565215 m, also below 1 mm. Acceleration, alpha and obstacle violations are zero for all 225 local trajectories (including all intermediate rounds). Final local discrete-dynamics discrepancies are below 5e-15. Final geometry errors by robot are:

| K | Robot 0 (m) | Robot 1 (m) | Robot 2 (m) |
| --- | ---: | ---: | ---: |
| 5 | 1.757660087e-07 | 7.589865451e-06 | 8.746345482e-06 |
| 10 | 2.249662606e-08 | 7.589865428e-06 | 8.746345482e-06 |
| 20 | 2.206304586e-08 | 7.58986545e-06 | 8.746345482e-06 |
| 40 | 4.708785362e-09 | 7.589865451e-06 | 8.746345482e-06 |

Every round and every final proposal has RTI statuses [0,0,0] and QP statuses [0,0,2], in robot order. There are no outer RTI failures. QP status 2 remains a maximum-iteration return at the unchanged 50-iteration limit for robot 2. QP status counts (0 / 2) are 10 / 5, 20 / 10, 40 / 20 and 80 / 40 for K=5,10,20,40. Across the complete sweep there are 150 QP successes and 75 maximum-iteration returns; these retain the existing acceptance policy and do not establish NLP optimality. Per-robot NLP residuals for every call are preserved in the diagnostic JSON files.

| K | Total serial controller update (ms) | Mean local solve/retrieval (ms) | Mean native solve (ms) |
| --- | ---: | ---: | ---: |
| 5 | 44.077584 | 1.444916 | 1.277394 |
| 10 | 112.459250 | 1.496654 | 1.317911 |
| 20 | 179.184708 | 1.481883 | 1.307675 |
| 40 | 351.484833 | 1.448317 | 1.277925 |

Controller update time includes the existing warm-start preparation, parameter loading, serial primal calls, diagnostics and ADMM operations. It excludes native creation, snapshot reconstruction/restoration, result writing and plant execution. Mean local solve time includes solution retrieval, matching the existing logging definition. These are single measurements per frozen budget, not a timing benchmark. K=10's measured 112.459250 ms exceeds the nominal 33.333333 ms interval.

The per-round histories below apply to every case through its budget, because their complete round arrays are identical prefixes. All entries use the original residual definitions; full-horizon quantities remain diagnostics. Geometry, constraint violations, statuses, NLP residuals and local timing for every round are available in `sweep_report.json` and the per-budget diagnostic files.

| Round | First velocity (m/s) | First angular (rad/s) | Full-horizon velocity (m/s) | Full-horizon angular (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.003724672504 | 0.001223880609 | 0.007372877063 | 0.001223880609 |
| 2 | 0.004426114905 | 0.001345349616 | 0.004944735023 | 0.001345349616 |
| 3 | 0.002922994677 | 0.0006589355187 | 0.002922994677 | 0.0006589355187 |
| 4 | 0.001997111137 | 0.0001868741606 | 0.001997111137 | 0.0001992475657 |
| 5 | 0.00315705266 | 0.0007178195069 | 0.00315705266 | 0.0007178195069 |
| 6 | 0.003364694474 | 0.0009035747771 | 0.003364694474 | 0.0009035747771 |
| 7 | 0.002830247683 | 0.0007901917097 | 0.002830247683 | 0.0007901917097 |
| 8 | 0.001932991215 | 0.0005253248087 | 0.001932991215 | 0.0005253248087 |
| 9 | 0.001022654699 | 0.0002492500033 | 0.001022654699 | 0.0002492500033 |
| 10 | 0.0003155473811 | 4.919709223e-05 | 0.000694908793 | 0.0002125631315 |
| 11 | 0.0001299037223 | 4.775191135e-05 | 0.0006575922094 | 0.0001877035926 |
| 12 | 0.0003356939172 | 5.938726001e-05 | 0.0005396591308 | 0.0001427757354 |
| 13 | 0.000417428658 | 2.389916527e-05 | 0.000417428658 | 9.344207056e-05 |
| 14 | 0.0005412980521 | 2.542555284e-05 | 0.0005412980521 | 7.523132086e-05 |
| 15 | 0.0007242965996 | 5.290442597e-05 | 0.0007242965996 | 6.269347452e-05 |
| 16 | 0.0008516249439 | 6.062337041e-05 | 0.0008516249439 | 6.062337041e-05 |
| 17 | 0.000928963504 | 5.029293857e-05 | 0.000928963504 | 5.029293857e-05 |
| 18 | 0.0009732382659 | 3.077327262e-05 | 0.0009732382659 | 3.637799724e-05 |
| 19 | 0.0009993562722 | 1.139304231e-05 | 0.0009993562722 | 3.229938296e-05 |
| 20 | 0.001022949935 | 2.021304156e-06 | 0.001022949935 | 2.857718878e-05 |
| 21 | 0.001051973527 | 7.875868317e-06 | 0.001051973527 | 2.990952705e-05 |
| 22 | 0.001047958694 | 7.729791048e-06 | 0.001047958694 | 3.344805243e-05 |
| 23 | 0.001019172361 | 4.517839404e-06 | 0.001019172361 | 3.41943335e-05 |
| 24 | 0.0009735633437 | 1.437173267e-06 | 0.0009735633437 | 3.300937702e-05 |
| 25 | 0.0009223550085 | 2.28481089e-06 | 0.0009223550085 | 3.087932544e-05 |
| 26 | 0.000857712597 | 2.411225767e-06 | 0.000857712597 | 2.859977009e-05 |
| 27 | 0.0007801436922 | 1.284064345e-06 | 0.0007801436922 | 2.664740252e-05 |
| 28 | 0.000695514035 | 1.126369636e-06 | 0.000695514035 | 2.515364108e-05 |
| 29 | 0.0006056743275 | 2.626214761e-06 | 0.0006056743275 | 2.400600262e-05 |
| 30 | 0.0005087670874 | 3.581285132e-06 | 0.0005164760292 | 2.297278591e-05 |
| 31 | 0.0004065872411 | 3.916755766e-06 | 0.0004759688998 | 2.181951454e-05 |
| 32 | 0.0003014828963 | 3.794480557e-06 | 0.0004306597132 | 2.038627689e-05 |
| 33 | 0.0001959612875 | 3.464598307e-06 | 0.0003812598613 | 1.861753042e-05 |
| 34 | 9.237159545e-05 | 3.143383937e-06 | 0.0003287535881 | 1.655197292e-05 |
| 35 | 2.055557575e-05 | 2.950837272e-06 | 0.0002742909006 | 1.4288744e-05 |
| 36 | 0.0001131721326 | 2.906931671e-06 | 0.000219058934 | 1.194747883e-05 |
| 37 | 0.0002007174243 | 2.964880269e-06 | 0.0002044264246 | 9.635623499e-06 |
| 38 | 0.0002805714085 | 3.055034111e-06 | 0.0002805714085 | 7.429847239e-06 |
| 39 | 0.0003515333938 | 3.119572758e-06 | 0.0003515333938 | 6.593602092e-06 |
| 40 | 0.0004126865546 | 3.129214983e-06 | 0.0004126865546 | 6.226335594e-06 |

Reproduction:

```bash
source .venv/acados_env.sh
python sweep_dnmpc_frozen_admm.py \
  --source logs/dnmpc_consensus_1mm_20step/n3_seed1234 \
  --output logs/dnmpc_frozen_step1_admm_sweep/n3_seed1234
```

The completed sweep used 225 local calls plus 15 reconstruction calls. An initial invocation reproduced K=5 but stopped while serializing a NumPy integer in the log writer; that invocation made 15 reconstruction and 15 K=5 calls, with zero plant updates. Only the diagnostic serialization was corrected before the complete invocation. No production controller or numerical setting changed. Timings above are from the completed invocation.

Artifacts are under `logs/dnmpc_frozen_step1_admm_sweep/n3_seed1234/`: `frozen_snapshot.npz`, `sweep_report.json`, `sweep_report.md`, one `K_<budget>_diagnostic.json` per budget (05,10,20,40) and the corresponding `_trajectories.npz` files. `verification.json` records independent checks of all 225 nonlinear geometry profiles, dynamics and bounds, 75 full-horizon residuals, first-input errors, ADMM dual evolution and the unchanged shifted warm starts. A read-only audit confirmed the native snapshot method for the current cold-QP options; personal source/output review verified its conclusions. The scalar step index is saved explicitly in the snapshot. A separate non-mutating preview check also confirms that K=10 and K=40 pass the integrated candidate checks, with geometry maxima 1.06249687e-05 and 1.37497337e-05 m respectively (`pure_preview_verification.json`); it executes no plant step. Production source hashes are recorded in the report and remained unchanged. K=10 is a result for this frozen state only; no subsequent physical update or mission test was run.

## Nominal and obstacle-perception rollout experiment

The fixed default is now **K_ADMM=20**. Each local primal still uses exactly one SQP_RTI call per ADMM round. First-input and full-horizon translational/angular ADMM residuals continue to be recorded, but consensus is no longer a hard execution gate. No controls or trajectories are averaged: robot 0's first load input supplies the integrated candidate load motion, exactly as before. Native status/local-feasibility checks and the plant's finite, geometry (<=1e-3 m), acceleration, alpha and obstacle checks remain. Other external checks retain their 1e-6 thresholds. The OCP, native solver settings, 1/30 s timestep, 45-interval horizon, smooth reference, planar tether geometry, weights, bounds, rho and warm-start logic are unchanged.

Obstacle perception is `c_hat_j = c_j + b`, with `eta_j=0`. A normalized two-dimensional Gaussian draw supplies a uniform random direction, and `--obstacle-bias` sets its magnitude in metres. The vector is drawn once per mission and held fixed. The NumPy SeedSequence stream uses the master seed, absolute mission offset and a separate perception stream ID; it does not consume the JAX mission-reset RNG. Thus offsets 0–19 under seed 1234 provide paired missions and identical bias directions across the five disturbance magnitudes. Both the vector and direction are saved, including the zero vector for nominal operation.

True centers remain in the plant, actual state snapshots, rendering and safety evaluation. The controller receives a separate state copy containing only perceived centers. Its environment view contains only agent/obstacle counts and agent/payload radii, with no plant state or mission generator from which true centers could be accessed. The native parameter-packet spy test verifies that every local stage packet receives the biased centers. No obstacle inflation, robustness margin, independent obstacle noise, wind or controller retuning is introduced.

Each mission stops at its first failed controller or physical candidate check. A multi-mission evaluator continues with later requested mission keys without resampling invalid resets or failed initial warm starts. Reports distinguish executed states, rejected candidates and initialization failures. Goal success means that at least one committed payload position reaches within 0.1 m of the goal; final and closest position distances and wrapped yaw error are also recorded. This position threshold follows the existing mission generator and changes no controller cost or reference.

True obstacle safety is evaluated at committed states and throughout each executed interval using `p(t)=p0+v0*t+0.5*a*t^2`. The clearance helper minimizes squared distance at interval endpoints and every real stationary time from its cubic derivative. It logs true collisions and minimum clearance without changing physical acceptance. Geometry maxima are sampled committed-state values; local predicted and rejected-candidate violations remain separate. Each mission also records acceleration/alpha violations, RTI/QP status counts, all ADMM residual histories, controller/local/native timing, true/perceived centers and its sampled bias vector.

The existing true-obstacle physical gate can reject an unsafe proposal before execution. Consequently, zero observed collisions in truncated rollouts is not evidence of an ungated disturbance safety threshold. Reports retain unsafe true-obstacle proposals, rejection counts, mission completion and observed duration alongside actual collisions.

The experiment driver first requests one 10 s nominal mission (300 steps at 30 Hz). Its nominal prerequisite requires completion without rejection, physical safety, and either goal success or at least a 10% reduction in initial payload goal distance. The 10% criterion defines meaningful progress for experiment staging only. If it passes, the driver runs the requested bias magnitudes 0,0.02,0.05,0.10,0.15 m for 20 paired missions each; it reuses the validated nominal mission as level-zero mission 0. Each remaining mission runs in its own process to keep detailed native logs from accumulating in memory. No disturbance mission is launched when nominal validation fails.

```bash
source .venv/acados_env.sh
python run_dnmpc_bias_experiment.py --seed 1234 \
  --output logs/dnmpc_perception_shift
```

For individual mission debugging, `test_dnmpc.py --obstacle-bias <meters>` exposes the same perception model. `--epi 20` requests consecutive reproducible mission keys and records each mission's outcome; the conditional experiment driver enforces the nominal prerequisite and paired sweep protocol.

### Nominal validation result

The requested seed-1234, offset-0, N=3, complete-graph mission with three obstacles, bias=0, wind=0 and K_ADMM=20 **did not complete**. It committed 75 updates (2.5 s of simulated motion), then rejected physical step 75, the 76th attempted update. No disturbance missions were launched. This failure to complete the nominal prerequisite prevents estimating a disturbance degradation level.

| Quantity | Result |
| --- | ---: |
| Requested duration / committed duration | 10 s / 2.5 s |
| Committed / rejected updates | 75 / 1 |
| Goal reached within 0.1 m | No |
| Initial / final / closest goal distance | 1.645453246 / 1.523745188 / 1.523745188 m |
| Goal-distance reduction | 0.121708058 m (7.3966%) |
| Observed true robot-obstacle collisions | 0 |
| Minimum true clearance over executed intervals | 0.0348388802 m |
| Maximum committed geometry error | 3.653595274e-05 m (0.036536 mm) |
| Committed acceleration / alpha violation | 0 / 0 |
| Unsafe true-obstacle proposals | 0 / 76 |
| Sampled common bias vector | [0, 0] m |

These safety observations cover only the executed 2.5 s. Both collision-rate aliases mark this truncated mission as censored. They do not establish safety over the requested 10 s or under nonzero bias.

The exact rejection was `controller_not_ready` with `final_local_primal_infeasible_or_failed`: robot 1 returned RTI status 4 (`ACADOS_QP_FAILURE`) and QP status 1 (`ACADOS_NAN_DETECTED`). Its first failure occurred in ADMM round 5 of physical step 75, after 33 QP iterations; rounds 6–20 returned the same failure after one QP iteration each. Final RTI statuses were [0,4,0], and QP statuses were [0,1,2]. Robot 1's returned state/control arrays remained finite and explicitly feasible, and repeated failure returns retained its prior finite trajectory. That returned feasibility does not make the native solve successful. The observed cause of rejection is the QP NaN status; the underlying numerical cause has not been established, and these results do not demonstrate hard-OCP infeasibility.

The rejected integrated candidate itself passed the physical checks: geometry error 3.400216112e-05 m, zero acceleration/alpha/obstacle violations, and minimum true endpoint clearance 0.0333048084 m. Consensus was not the rejection reason. In particular, the last committed update had first-input velocity disagreement 0.00109709162 m/s, above the former 0.001 m/s gate, and was accepted with physical checks satisfied.

There were 4,560 local calls over 1,520 ADMM rounds. Outer RTI status counts were 4,544 successes and 16 QP failures. QP counts were 2,478 status-0 successes, 16 status-1 NaN returns, and 2,066 status-2 maximum-iteration returns. The unchanged policy permits an outer-successful, explicitly feasible RTI result with QP status 2; these returns do not prove NLP convergence. All 228 final local trajectories were explicitly feasible. Across intermediate rounds, 4,556/4,560 local trajectories were feasible (99.9123%); four early geometry violations had a maximum of 0.026524182 m and were not executed. Maximum final-local geometry error was 3.653595274e-05 m. Predicted acceleration, alpha and obstacle violations were zero throughout; maximum predicted dynamics error was 4.941180798e-11.

Residuals remain diagnostics. The maxima of first-input velocity and angular disagreement over the 76 final candidates were 0.00118436638 m/s and 0.000481551783 rad/s, respectively, both at the rejected update. Full-horizon final-round residuals were:

| Diagnostic | Mean over attempted updates | Maximum / last update |
| --- | ---: | ---: |
| Translational | 0.000937768391 m/s | 0.00169628116 m/s |
| Angular | 0.0000750463932 rad/s | 0.000806990513 rad/s |
| Combined | 0.000940795013 | 0.00187845774 |

Final NLP residuals at the rejected update are shown below in native order `(stationarity, equality, inequality, complementarity)`; every call's residuals and per-round ADMM history are preserved in the diagnostic JSON.

| Robot | Stationarity | Equality | Inequality | Complementarity |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4.970602482e-04 | 1.332267630e-15 | 2.614422012e-10 | 1.000000004e-08 |
| 1 | 7.989406443e-01 | 8.881784197e-16 | 9.568419667e-07 | 8.833878662e-04 |
| 2 | 9.203518729e-04 | 8.881784197e-16 | 3.765927015e-10 | 5.451006244e-08 |

| Timing scope | Mean | Maximum |
| --- | ---: | ---: |
| Serial controller update, 76 attempts | 225.125355 ms | 371.319500 ms |
| Local solve and solution retrieval, 4,560 calls | 1.993834 ms | 8.855333 ms |
| Native solve, 4,560 calls | 1.786635 ms | 8.521458 ms |

Controller timing includes warm-start preparation, parameter loading, serial local solves, diagnostics and ADMM operations; it excludes creation, plant execution and output writing. The measured serial update time exceeds the 33.333333 ms plant interval. The rollout uses a 30 Hz simulation timestep, but this implementation has not demonstrated real-time 30 Hz execution.

Artifacts are in `logs/dnmpc_perception_shift/`: `experiment_report.json` records the failed nominal prerequisite and an empty disturbance-level list. `nominal_seed1234/` contains `statistics.json`, `metadata.json`, `episode_diag.json`, `episode_0000.npz`, `episode_summary.csv`, `console.log` and the validation plot. `offline_verification.json` independently checks all 4,560 local geometry/dynamics/bound profiles, all 1,520 round residuals and all 75 exact plant integrations, true swept clearance and goal progress, without any additional native calls. A reporting-only audit subsequently added swept proposal clearances and corrected collision censoring; it changed no trajectories or execution decisions.

Personal review of the actual delegated files and current-task diffs confirmed the true/perceived center separation and unchanged OCP, solver, geometry, reference and warm-start implementation. All 37 unit tests pass, including biased native-packet capture, asymmetric load-input execution without averaging, exact swept clearance, truncated-collision censoring, and conditional sweep scheduling. Syntax and whitespace checks pass. The remaining limitation is the nominal QP failure, so neither the 100-mission safety/success aggregate nor an approximate bias degradation threshold is available. No tuning or additional rollout was performed after this nominal failure.
