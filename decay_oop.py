"""练习 1.7：放射性衰变的面向对象实现与误差分析。

分析题：
1. error = numerical - exact 恒为负：Euler 递推给出 (1 - dt/tau)^n，
   对 0 < dt/tau < 1 有 (1 - dt/tau)^n < e^{-n dt/tau}，故数值解始终低于解析解。
2. 右图 log-log 斜率约为 1，说明最大绝对误差与 dt 成正比，Euler 是一阶方法；
   若改用精确递推 N_{k+1} = N_k e^{-dt/tau}，误差降为舍入量级，图上不再有斜率 1 的直线。
3. dt=2.5 s > 2 tau 时 |1 - dt/tau| = 1.5 > 1，递推因子绝对值大于 1，
   数值解按 (-1.5)^n 交替放大，迅速发散；这是 Euler 法的稳定性边界。
"""

from __future__ import annotations

import sitecustomize  # noqa: F401  — local module

import numpy as np


class Decay:
    """放射性衰变的 Euler 数值解与解析解。"""

    def __init__(self, N0: float, tau: float, dt: float, t_max: float) -> None:
        self.N0 = N0
        self.tau = tau
        self.dt = dt
        self.t_max = t_max

    @property
    def N0(self) -> float:
        """初始核数。"""
        return self._N0

    @N0.setter
    def N0(self, value: float) -> None:
        if not value > 0:
            raise ValueError("N0 必须为正数")
        self._N0 = float(value)

    @property
    def tau(self) -> float:
        """时间常数 tau / s。"""
        return self._tau

    @tau.setter
    def tau(self, value: float) -> None:
        if not value > 0:
            raise ValueError("tau 必须为正数")
        self._tau = float(value)

    @property
    def dt(self) -> float:
        """Euler 步长 dt / s。"""
        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        if not value > 0:
            raise ValueError("dt 必须为正数")
        self._dt = float(value)

    @property
    def t_max(self) -> float:
        """模拟总时长 / s。"""
        return self._t_max

    @t_max.setter
    def t_max(self, value: float) -> None:
        if not value > 0:
            raise ValueError("t_max 必须为正数")
        self._t_max = float(value)

    @property
    def t(self) -> np.ndarray:
        """与 Euler 取值点严格对齐的时间网格 t_k = k dt。"""
        steps = int(np.floor(self.t_max / self.dt + 1e-9)) + 1
        return np.arange(steps, dtype=float) * self.dt

    @property
    def numerical(self) -> np.ndarray:
        """Euler 数值解（现算，不缓存）。"""
        grid = self.t
        values = np.empty_like(grid)
        values[0] = self.N0
        for index in range(grid.size - 1):
            values[index + 1] = self.euler_step(values[index], self.dt, self.tau)
        return values

    @property
    def exact(self) -> np.ndarray:
        """解析解 N0 exp(-t/tau)，取样点与 d.t 相同（现算，不缓存）。"""
        return self.N0 * np.exp(-self.t / self.tau)

    @property
    def error(self) -> np.ndarray:
        """带符号的逐点误差 numerical - exact（现算，不缓存）。"""
        return self.numerical - self.exact

    @property
    def max_abs_error(self) -> float:
        """最大绝对误差（现算，不缓存）。"""
        return float(np.max(np.abs(self.error)))

    @property
    def half_life(self) -> float:
        """半衰期 tau ln 2 / s。"""
        return float(self.tau * np.log(2.0))

    @staticmethod
    def euler_step(n: float, dt: float, tau: float) -> float:
        """单步 Euler：n - n dt / tau。"""
        return n - n * dt / tau

    @classmethod
    def from_half_life(cls, N0: float, t_half: float, dt: float, t_max: float) -> "Decay":
        """用半衰期构造：tau = t_half / ln 2。"""
        return cls(N0, t_half / np.log(2.0), dt, t_max)


def build_figure() -> str:
    """生成 1x3 误差分析图并保存为 decay_oop.png。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Source Han Sans SC", "Noto Sans CJK SC", "PingFang SC",
        "Heiti SC", "SimHei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    N0, tau, t_max = 100.0, 1.0, 5.0
    base = Decay(N0, tau, 0.2, t_max)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(base.t, base.exact, label="解析解 $N_0 e^{-t/\\tau}$")
    axes[0].scatter(base.t, base.numerical, s=18, label="Euler 数值解 ($\\Delta t=0.2$ s)")
    axes[0].set_xlabel("时间 $t$/s")
    axes[0].set_ylabel("剩余核数 $N$")
    axes[0].legend()

    axes[1].plot(base.t, base.error, color="C3")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("时间 $t$/s")
    axes[1].set_ylabel("误差 $N_{num}-N_{exact}$")
    axes[1].set_title(f"逐点误差（最大绝对误差 {base.max_abs_error:.4f}）")

    steps = [0.5, 0.25, 0.2, 0.1, 0.05, 0.025]
    errors = [Decay(N0, tau, step, t_max).max_abs_error for step in steps]
    slope = float(np.polyfit(np.log(steps), np.log(errors), 1)[0])
    axes[2].loglog(steps, errors, "o-", label="最大绝对误差")
    reference = np.array(steps, dtype=float)
    axes[2].loglog(reference, reference / reference[0] * errors[0], "k--", label="斜率 1 参考线")
    axes[2].set_xlabel("步长 $\\Delta t$/s")
    axes[2].set_ylabel("最大绝对误差")
    axes[2].set_title(f"误差收敛阶（拟合斜率 {slope:.3f}）")
    axes[2].legend()

    figure.tight_layout()
    figure.savefig("decay_oop.png", dpi=160)
    plt.close(figure)
    return "decay_oop.png"


def main() -> int:
    base = Decay(100.0, 1.0, 0.2, 5.0)
    print(f"half_life = {base.half_life:.4f} s")
    print(f"N(5), dt=0.2 = {base.numerical[-1]:.4f}")
    print(f"exact(5) = {base.exact[-1]:.4f}")
    print(f"max_abs_error, dt=0.2 = {base.max_abs_error:.4f}")
    base.dt = 0.05
    print(f"max_abs_error, dt=0.05 = {base.max_abs_error:.4f}")
    print(f"saved {build_figure()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
