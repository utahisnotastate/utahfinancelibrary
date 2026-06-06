# 用語集

> **言語:** [English](../glossary.md) · [Eesti](../et/glossary.md) · [Русский](../ru/glossary.md) · 日本語

> 用語は原語のまま残しています（コードや API 名に対応するため）。定義を日本語に訳しています。

| 用語 | 定義 |
|------|------|
| **Adelic Clearinghouse Bypass** | 局所大域の支払能力を検証し、チェックが通ればモデル化担保ゼロを許す決済経路 |
| **AllocationIntent** | 利回り摩擦を減らすため会場間で資本を移す指示 |
| **Atomic settlement** | 取引執行と最終台帳更新が 1 ステップで起こる |
| **AutonomousAuditor** | 会場横断の資本流出を検出するエンジン |
| **Capital drag** | 資産を最適でない会場に置くことで失われる利回り |
| **Hasse-Minkowski verifier** | 決済前に実体 + 法体の支払能力チェックを行うコンポーネント |
| **Harvest** | 決済エンジンが処理する実現利回りイベント |
| **Humanitarian abundance matrix** | インパクト配分用に設定されたウォレット/経路（既定 5.7%） |
| **k-sunflower routing** | 流動性ノードをサイズ ≤ k の互いに素な花弁に分割 |
| **Omnibus audit** | ブール値チェックの束: 什一、スペクトル剛性、互いに素なルーティング |
| **Orthogonal phase-lock** | PINN 隠れ層の活性化 $\sin(x)e^{-x^2/2}$ |
| **PINN** | 物理情報付きニューラルネット — 損失に構造的物理ペナルティを含む |
| **Protocol tithe** | 主権議定書への不変の 2.3% 配分 |
| **SettlementInstruction** | 1 件の支払行（ウォレット、金額、経路ラベル） |
| **Sovereign vault** | しきい値署名された意図カストディ層 |
| **Sunflower petal** | ルーティング位相における 1 つの互いに素な資本ノード集合 |
| **UCE** | Utah Container Engine — Utahfile で参照される外部デプロイランタイム |
| **Utahfile** | サービス、フック、エンジン設定を記述する YAML マニフェスト |
| **Verification lattice** | `InvarianceValidationLattice` — 実行時の数学的チェック |
| **Wave telemetry** | 市場ティックに対する有界リングバッファ統計 |
| **Zero collateral** | アデール検証後の `required_collateral == 0` |
| **Persistent homology** | データ形状の多スケール位相要約（Betti 数、バーコード） |
| **Betti numbers** | 複体の $b_0$ 連結成分、$b_1$ ループ、$b_2$ 空洞 |
| **Topological Risk Parity (TRP)** | 位相的伝染負荷に反比例して重み付けされた配分 |
| **Vietoris-Rips complex** | 半径内の点を結んで構築する単体複体 |
| **Wasserstein barcode distance** | 持続性図の間の最適輸送距離 |
| **Ricci flow** | メトリックを定曲率へ進化させる PDE $\partial_t g_{ij} = -2R_{ij}$ |
| **Ricci flow covariance** | リーマン・メトリックとして進化させてノイズ除去された共分散 |
| **Navier-Stokes routing** | 非圧縮（質量保存）流体流としてモデル化したリバランス |
| **Velocity field** | ルーティングソルバーが返す連続的な資本フロー率 |
| **Spectral CVaR veto** | 損失作用素のスペクトル半径が壁を破ると発火する停止トリガー |
| **Symplectic Veto** | スペクトル CVaR 監督が発火させる執行停止アクション |
| **Dirichlet boundary conditions** | 領域端で固有関数を固定値に留める |
| **Christoffel symbols** | $\Gamma^k_{ij}$、メトリックの接続係数 |
| **Riemann tensor** | $R^l{}_{ijk}$、リーマン多様体の完全な曲率 |
| **Ricci tensor** | $R_{jk}$、Riemann のトレース。リッチフローを駆動 |
| **Scalar curvature** | $R = g^{jk}R_{jk}$、一数の曲率 |
| **Laplace-Beltrami operator** | $\Delta_M$、リーマン多様体上のラプラシアン |
| **Principal eigenvalue** | 生成作用素の最小 Dirichlet 固有値 $\lambda_0$ |
| **Feynman-Kac bound** | $\mathbb{P}(\sup\text{DD}>\mathcal D_{max})\le Ce^{-\lambda_0 T}$ |
| **Volume-normalised Ricci flow** | 体積を固定する項を持つリッチフロー。定曲率へ収束 |
| **Betti-Number Divergence Test** | クラッシュ診断: 生 $b_0\to1$ 対 デトーン $b_0>1$ |
| **Detoning** | 相関行列から支配的な市場/システミック固有モードを除去 |
| **Quadratic covariation** | $\langle X_i,X_j\rangle_t$、二乗増分和のドリフト非依存なパス単位極限 |
| **Pathwise metric tensor** | $g_{ij}(t)=\tfrac{d}{dt}\langle X_i,X_j\rangle_t$、推定ではなく測定されたメトリック |
| **Realized covariation** | 対数収益増分の外積和。$dt\to0$ で $\langle X\rangle_t$ に収束 |
| **TSRV** | Two-Scale Realized Covariance: 生ティック向けのノイズ頑健な共変動推定 |
| **Tick observer** | ティックストリームからの $g_{ij}(t)$ のオンライン・ラグなし観測器 (`tick_observer.py`) |
| **Holographic LOB embedding** | AdS/CFT 風の板情報深さの双曲的重み付けを有界圧力特徴量へ |
| **Von Neumann entropy** | $S(\rho)=-\operatorname{tr}(\rho\log\rho)$。ここでは正規化共分散の分散/集中の尺度 |
| **Effective number of bets** | $e^{S(\rho)}$、独立な能動的リスク方向の数 |
| **Matrix Product State (MPS)** | 状態ベクトルのテンソルネットワーク分解。ボンドエントロピー = カット間の絡み合い |
| **Malliavin derivative** | パス汎関数のノイズ摂動への感度 $D_t F$。グリークスに使用 |
| **Skorokhod integral** | 先読み的（非適合）確率積分。Malliavin 微分の随伴 |
| **Protocol yield split** | 正の利回りの透明・設定可能なネット/人道/什一ルーティング (`protocol_economics.py`) |
| **Universal tithe** | 正の利回りの透明・設定可能・除去可能な 10.0% 人道 + 2.3% 議定書ルーティング (`enforce_universal_tithe`) |
| **Jarzynski equality** | $\langle e^{-\beta W}\rangle = e^{-\beta\Delta F}$。非平衡仕事から平衡自由エネルギー差を復元 |
| **Crooks fluctuation theorem** | $P_F(W)/P_R(-W)=e^{\beta(W-\Delta F)}$。順/逆の仕事分布を関係づける |
| **Dissipated work** | $\langle W\rangle-\Delta F\ge0$。$k_BT\,D_{\mathrm{KL}}(P_F\|P_R)$ に等しい（不可逆性スケール） |
| **Koopman operator** | 非線形系の観測量に作用する線形作用素 $(\mathcal K g)(x)=g(F(x))$ |
| **EDMD** | Extended Dynamic Mode Decomposition: $\mathcal K$ の最小二乗有限近似 $K=G_yG_x^+$ |
| **Artin braid group** | $B_n$、交差 $\sigma_i$ で生成される $n$ 本ひもの組みひも群 |
| **Temperley-Lieb algebra** | $TL_n(\delta)$ 平面図式代数、$e_i^2=\delta e_i$。組みひもスケイン写像の像 |
| **Kauffman bracket** | $\langle L\rangle$、$\sigma_i\mapsto A\mathbf1+A^{-1}e_i$ による正則イソトピー不変量 |
| **Jones polynomial** | $t=A^{-4}$ での $V_L(t)=(-A^3)^{-w}\langle L\rangle$。絡み目のイソトピー不変量 |
| **Cross-impact ordering** | 執行順序にわたる非対称インパクトコスト $\sum_{a\,\text{before}\,b}L_{ab}$ の最小化 |
