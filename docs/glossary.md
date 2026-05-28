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
