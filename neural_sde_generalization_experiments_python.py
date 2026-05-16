import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Device configuration
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ============================================================
# Neural SDE Model
# ============================================================
class NeuralSDE(nn.Module):
    """
    Neural Stochastic Differential Equation (Neural SDE)
    implemented using Euler-Maruyama discretization.
    """

    def __init__(self, d, m, num_classes, hidden=64):
        super().__init__()

        self.m = m

        # Drift components
        self.f_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, d)
            )
            for _ in range(m)
        ])

        # Learnable coefficients
        self.theta = nn.Parameter(torch.randn(m))

        # Final classifier
        self.readout = nn.Linear(d, num_classes)

    def forward(self, x, T=1.0, steps=15, eps=0.02):
        """
        Euler-Maruyama simulation of the Neural SDE.
        """

        dt = T / steps
        h = x

        for _ in range(steps):

            # Drift term
            drift = 0
            for i in range(self.m):
                drift += self.theta[i] * self.f_list[i](h)

            # Brownian noise
            noise = torch.randn_like(h) * np.sqrt(dt)

            # Euler-Maruyama update
            h = h + drift * dt + eps * noise

        return self.readout(h)


# ============================================================
# Dataset Loader
# ============================================================
def get_dataset(name="MNIST"):
    """
    Load dataset.
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if name == "MNIST":

        trainset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        testset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        d = 28 * 28

    else:
        raise ValueError("Only MNIST dataset is supported.")

    return trainset, testset, d


# ============================================================
# Random subset selection
# ============================================================
def get_subset(dataset, n):

    indices = torch.randperm(len(dataset))[:n]

    subset = torch.utils.data.Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=n,
        shuffle=True
    )

    X, y = next(iter(loader))

    X = X.view(n, -1).to(device)
    y = y.to(device)

    return X, y


# ============================================================
# Training
# ============================================================
def train_model(model, X, y, T, epochs=50):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = model(X, T)

        loss = loss_fn(outputs, y)

        loss.backward()

        optimizer.step()

    return model


# ============================================================
# Generalization Gap
# ============================================================
def generalization_gap(model, X_train, y_train, X_test, y_test, T):

    loss_fn = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():

        train_loss = loss_fn(
            model(X_train, T),
            y_train
        ).item()

        test_loss = loss_fn(
            model(X_test, T),
            y_test
        ).item()

    return test_loss - train_loss


# ============================================================
# Experiment Runner
# ============================================================
def run_experiment(dataset_name, param_list, mode, seeds=3):

    trainset, testset, d = get_dataset(dataset_name)

    results = []

    for val in param_list:

        gaps = []

        for seed in range(seeds):

            torch.manual_seed(seed)
            np.random.seed(seed)

            # ------------------------------------------------
            # Experiment configurations
            # ------------------------------------------------
            if mode == "n":
                n = val
                m = 10
                T = 1.0

            elif mode == "m":
                n = 500
                m = val
                T = 1.0

            elif mode == "T":
                n = 500
                m = 10
                T = val

            else:
                raise ValueError("Mode must be one of: n, m, T")

            # ------------------------------------------------
            # Data
            # ------------------------------------------------
            X_train, y_train = get_subset(trainset, n)
            X_test, y_test = get_subset(testset, 1000)

            # ------------------------------------------------
            # Model
            # ------------------------------------------------
            model = NeuralSDE(
                d=d,
                m=m,
                num_classes=10
            ).to(device)

            # ------------------------------------------------
            # Training
            # ------------------------------------------------
            model = train_model(model, X_train, y_train, T)

            # ------------------------------------------------
            # Generalization gap
            # ------------------------------------------------
            gap = generalization_gap(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                T
            )

            gaps.append(gap)

        avg_gap = np.mean(gaps)

        results.append(avg_gap)

        print(f"{dataset_name} | {mode}={val}, gap={avg_gap:.4f}")

    return results


# ============================================================
# Plotting utilities
# ============================================================
def plot_results(x, y, title, save_path=None, log=False):

    plt.figure(figsize=(6, 4))

    plt.plot(x, y, marker='o')

    if log:
        plt.xscale("log")
        plt.yscale("log")

    plt.title(title)
    plt.xlabel("Parameter")
    plt.ylabel("Generalization Gap")

    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


# ============================================================
# Main Experiment Pipeline
# ============================================================
if __name__ == "__main__":

    dataset = "MNIST"

    # ========================================================
    # Scaling with sample size n
    # ========================================================
    ns = [100, 200, 500, 1000, 2000]

    res_n = run_experiment(dataset, ns, mode="n")

    plot_results(
        ns,
        res_n,
        title="Generalization Gap vs Sample Size n",
        save_path="images/gen_gap_n.png",
        log=True
    )

    # ========================================================
    # Scaling with model dimension m
    # ========================================================
    ms = [2, 5, 10, 20, 40]

    res_m = run_experiment(dataset, ms, mode="m")

    plot_results(
        ms,
        res_m,
        title="Generalization Gap vs Model Dimension m",
        save_path="images/gen_gap_m.png"
    )

    # ========================================================
    # Scaling with time horizon T
    # ========================================================
    Ts = [0.5, 1.0, 1.5, 2.0, 3.0]

    res_T = run_experiment(dataset, Ts, mode="T")

    plot_results(
        Ts,
        res_T,
        title="Generalization Gap vs Time Horizon T",
        save_path="images/gen_gap_T.png"
    )

    print("\nAll experiments completed successfully.")
