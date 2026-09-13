"""3.2 Empirical: Visualizing SGD Behavior on 2D quadratics."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

START, LR, STEPS = (2.0, 1.5), 0.1, 50
MOMENTA = (0.0, 0.9)


def bowl(p):
    return (p ** 2).sum()  # f = x^2 + y^2, minimum at origin


def cap(p):
    return -(p ** 2).sum()  # f = -x^2 - y^2, maximum at origin


def run(f, **opt_kw):
    """Optimize f with torch.optim.SGD from START; return (STEPS+1, 2) trajectory."""
    p = torch.tensor(START, requires_grad=True)
    opt = torch.optim.SGD([p], lr=LR, **opt_kw)
    traj = [p.detach().clone()]
    for _ in range(STEPS):
        opt.zero_grad()
        f(p).backward()
        opt.step()
        traj.append(p.detach().clone())
    return torch.stack(traj).numpy()


def plot(f, sign, fname, title, **opt_kw):
    """Trajectories for each momentum on the contour plot of f; returns them."""
    g = np.linspace(-2.6, 2.6, 200)
    X, Y = np.meshgrid(g, g)
    fig, axes = plt.subplots(1, len(MOMENTA), figsize=(10, 4.8), layout="constrained")
    trajs = []
    for ax, m in zip(axes, MOMENTA):
        traj = run(f, momentum=m, **opt_kw)
        trajs.append(traj)
        ax.contour(X, Y, sign * (X ** 2 + Y ** 2), levels=15, cmap="Greys", linewidths=0.8)
        ax.plot(traj[:, 0], traj[:, 1], "-", c="grey", lw=0.8)
        sc = ax.scatter(*traj.T, c=np.arange(len(traj)), cmap="viridis", s=14, zorder=3)
        ax.plot(0, 0, "r*", ms=12)
        ax.set(aspect="equal", xlabel="x", ylabel="y", title=f"momentum={m}")
        print(f"  {title} | momentum={m}: ||p_{STEPS}|| = {np.linalg.norm(traj[-1]):.2e}")
    fig.colorbar(sc, ax=axes, label="step")
    fig.suptitle(title)
    fig.savefig(fname, dpi=150)
    return trajs


def main():
    a = plot(bowl, 1, "3.2_a_momentum.png", "(a) minimize x²+y²")
    plot(bowl, 1, "3.2_b_weight_decay.png", "(b) minimize x²+y², weight_decay=0.1", weight_decay=0.1)
    c = plot(cap, -1, "3.2_c_maximize.png", "(c) maximize=True on -x²-y²", maximize=True)
    print(f"  max |(c) - (a)| = {max(np.abs(x - y).max() for x, y in zip(a, c)):.1e}")


if __name__ == "__main__":
    main()
