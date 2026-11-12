# 练习 1.7 自查结果

运行 `python decay_oop.py`（N0=100, tau=1 s, t_max=5 s）实测：

| 检查项 | 实测 |
|---|---|
| `d.half_life` | 0.6931 s |
| `N(5)`, dt=0.2 | 0.3778（解析 0.6738） |
| `max_abs_error`, dt=0.2 | 4.0199 |
| `max_abs_error`, dt=0.05 | 0.9394 |
| log-log 拟合斜率 | 约 1.0 |
| dt=2.5 时 | 数值解发散 |

图：`decay_oop.png`（1×3 子图，dpi=160）。
