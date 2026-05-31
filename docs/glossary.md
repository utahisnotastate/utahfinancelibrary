# Glossary

| Term | Definition |
|------|------------|
| **Adelic Clearinghouse Bypass** | Settlement path that verifies local-global solvency and allows zero modeled collateral when checks pass |
| **AllocationIntent** | Instruction to move capital between venues to reduce yield drag |
| **Atomic settlement** | Trade execution and final ledger update occur in one step |
| **AutonomousAuditor** | Engine that detects cross-venue capital leakage |
| **Capital drag** | Lost yield from holding assets at suboptimal venues |
| **Hasse-Minkowski verifier** | Component implementing real + modular solvency checks before settlement |
| **Harvest** | Realized yield event processed by settlement engine |
| **Humanitarian abundance matrix** | Configured wallet/route for impact allocation (default 5.7%) |
| **k-sunflower routing** | Partitioning liquidity nodes into disjoint petals of size ≤ k |
| **Omnibus audit** | Bundle of boolean checks: tithe, spectral rigidity, disjoint routing |
| **Orthogonal phase-lock** | Activation $\sin(x)e^{-x^2/2}$ used in PINN hidden layer |
| **PINN** | Physics-Informed Neural Network — loss includes structural physics penalty |
| **Protocol tithe** | Immutable 2.3% allocation to sovereign protocol |
| **SettlementInstruction** | Single payout row (wallet, amount, route label) |
| **Sovereign vault** | Threshold-signed intent custody layer |
| **Sunflower petal** | One disjoint set of capital nodes in routing topology |
| **UCE** | Utah Container Engine — external deployment runtime referenced in Utahfile |
| **Utahfile** | YAML manifest describing services, hooks, and engine config |
| **Verification lattice** | `InvarianceValidationLattice` — mathematical runtime checks |
| **Wave telemetry** | Bounded ring-buffer statistics on market ticks |
| **Zero collateral** | `required_collateral == 0` after adelic verification |
| **Persistent homology** | Multi-scale topological summary (Betti numbers, barcodes) of data shape |
| **Betti numbers** | $b_0$ components, $b_1$ loops, $b_2$ voids of a complex |
| **Topological Risk Parity (TRP)** | Allocation inversely weighted by topological contagion load |
| **Vietoris-Rips complex** | Simplicial complex built by connecting points within a radius |
| **Wasserstein barcode distance** | Optimal-transport distance between persistence diagrams |
| **Ricci flow** | PDE $\partial_t g_{ij} = -2R_{ij}$ evolving a metric toward constant curvature |
| **Ricci flow covariance** | Covariance denoised by evolving it as a Riemannian metric |
| **Navier-Stokes routing** | Rebalancing modeled as incompressible (mass-conserving) fluid flow |
| **Velocity field** | Continuous capital-flow rates returned by the routing solver |
| **Spectral CVaR veto** | Halt trigger when a loss operator's spectral radius breaches a wall |
| **Symplectic Veto** | The execution-halt action fired by the spectral CVaR supervisor |
| **Dirichlet boundary conditions** | Pin eigenfunctions to fixed values at the domain edge |
| **Christoffel symbols** | $\Gamma^k_{ij}$, connection coefficients of a metric |
| **Riemann tensor** | $R^l{}_{ijk}$, full curvature of a Riemannian manifold |
| **Ricci tensor** | $R_{jk}$, trace of Riemann; drives Ricci flow |
| **Scalar curvature** | $R = g^{jk}R_{jk}$, single-number curvature |
| **Laplace-Beltrami operator** | $\Delta_M$, the Laplacian on a Riemannian manifold |
| **Principal eigenvalue** | Smallest Dirichlet eigenvalue $\lambda_0$ of the generator |
| **Feynman-Kac bound** | $\mathbb{P}(\sup\text{DD}>\mathcal D_{max})\le Ce^{-\lambda_0 T}$ |
| **Volume-normalised Ricci flow** | Ricci flow with a term holding volume fixed, converging to constant curvature |
| **Betti-Number Divergence Test** | Crash diagnostic: raw $b_0\to1$ vs detoned $b_0>1$ |
| **Detoning** | Removing dominant market/systemic eigenmodes from a correlation matrix |
| **Quadratic covariation** | $\langle X_i,X_j\rangle_t$, drift-invariant pathwise limit of summed squared increments |
| **Pathwise metric tensor** | $g_{ij}(t)=\tfrac{d}{dt}\langle X_i,X_j\rangle_t$, the measured (not estimated) metric |
| **Realized covariation** | Sum of outer products of log-return increments; converges to $\langle X\rangle_t$ as $dt\to0$ |
| **TSRV** | Two-Scale Realized Covariance: noise-robust covariation estimator for raw ticks |
| **Tick observer** | Online lag-free observer of $g_{ij}(t)$ from the tick stream (`tick_observer.py`) |
| **Holographic LOB embedding** | AdS/CFT-style hyperbolic weighting of order-book depth into a bounded pressure feature |
| **Von Neumann entropy** | $S(\rho)=-\operatorname{tr}(\rho\log\rho)$; here a diversification/concentration measure of normalised covariance |
| **Effective number of bets** | $e^{S(\rho)}$, count of independent active risk directions |
| **Matrix Product State (MPS)** | Tensor-network factorization of a state vector; bond entropies = entanglement across cuts |
| **Malliavin derivative** | Sensitivity $D_t F$ of a path functional to noise perturbation; used for Greeks |
| **Skorokhod integral** | Anticipating (non-adapted) stochastic integral; adjoint of the Malliavin derivative |
| **Protocol yield split** | Transparent, configurable net/humanitarian/tithe routing of positive yield (`protocol_economics.py`) |
