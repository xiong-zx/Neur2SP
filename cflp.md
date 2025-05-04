# Capacitated Facility Location Problem (CFLP)

## Extensive form of CFLP

- `x_i`：二元变量，表示设施 `i` 是否开设。
- `y_{i}_{j}_{s}`：连续或二元变量，表示在场景 `s` 下客户 `j` 是否由设施 `i` 提供服务。
- `z_{j}_{s}`：连续或二元变量，表示在场景 `s` 下客户 `j` 的补救变量，用于满足需求。
- `f_i`：非负实数，表示设施 `i` 的固定成本。
- `c_{ij}`：非负实数，表示客户 `j` 由设施 `i` 提供服务的费用。
- `r_j`：非负实数，表示客户 `j` 的补救费用。
- `d_j`：非负实数，表示客户 `j` 的需求。
- `s_i`：非负实数，表示设施 `i` 提供服务的能力。

$$
\begin{aligned}
    \min & \sum_{i} f_i x_i + \frac{1}{|S|} \sum_{s}\left( \sum_{i,j} c_{ij} y_{i,j,s} + \sum_{j} r_j z_{j,s} \right) \\
    \text{s.t.} & \sum_{i} y_{i,j,s} + z_{j,s} \geq 1 \quad \forall j, \forall s \\
    & \sum_{j} d_j y_{i,j,s} \leq s_i x_i \quad \forall i, \forall s \\
    & y_{i,j,s} \leq x_i \quad \forall i, \forall j, \forall s
\end{aligned}
$$

## Deterministic seconded-stage problem

$$
\begin{aligned}
    \min & \sum_{i} f_i x_i +  \sum_{i,j} c_{ij} y_{i,j} + \sum_{j} r_j z_{j} \\
    \text{s.t.} & \sum_{i} y_{i,j} + z_{j} \geq 1 \quad \forall j \\
    & \sum_{j} d_j y_{i,j} \leq s_i x_i \quad \forall i \\
    & y_{i,j} \leq x_i \quad \forall i, \forall j
\end{aligned}
$$
