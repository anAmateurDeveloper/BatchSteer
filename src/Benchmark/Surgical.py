import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score

class LogisticModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def demographic_parity(preds, sensitive):
    p1 = preds[sensitive == 1].mean()
    p0 = preds[sensitive == 0].mean()
    return abs(p1 - p0)


def generate_surgical_points(model, X, A, target_gap=0.02, n_points=10, lr=0.05):
    """
    Performs direct gradient manipulation to create a tiny batch
    of synthetic examples that correct fairness imbalance.
    """
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)

    # start surgical points near group-0 mean
    init_mean = X[A == 0].mean(0)
    antidote = init_mean.repeat(n_points, 1).clone().detach().requires_grad_(True)

    optimizer = optim.Adam([antidote], lr=lr)

    for step in range(200):
        optimizer.zero_grad()

        preds = model(X)
        fairness = demographic_parity(preds.detach(), A)

        # encourage antidotes to push predictions for group 0 *upwards* (if they are disadvantaged)
        new_preds = model(antidote)
        loss = fairness - new_preds.mean()  # gradient surgery

        loss.backward()
        optimizer.step()

        if fairness < target_gap:
            break

    return antidote.detach().cpu().numpy(), np.ones(n_points)  # label = 1

def surgical_fairness_fix(X, y, A, target_gap=0.02, inject=10):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = LogisticModel(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=0.05)

    # Stage 1 — Train original
    for _ in range(200):
        opt.zero_grad()
        p = model(X).squeeze()
        loss = nn.BCELoss()(p, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        init_preds = model(X).detach().numpy()
        init_gap = demographic_parity(init_preds, A.numpy()).item()
        print(f"Initial fairness gap: {init_gap:.4f}")

    # Stage 2 — Generate surgical antidote examples
    X_surg, y_surg = generate_surgical_points(model, X.numpy(), A.numpy(),
                                              target_gap=target_gap,
                                              n_points=inject)

    # Stage 3 — Surgical update: ONLY top-layer bias + relevant weights
    for param in model.linear.parameters():
        param.requires_grad_(False)
    model.linear.bias.requires_grad_(True)

    X_full = torch.cat([X, torch.tensor(X_surg, dtype=torch.float32)])
    y_full = torch.cat([y, torch.tensor(y_surg, dtype=torch.float32)])

    opt2 = optim.Adam([model.linear.bias], lr=0.05)
    for _ in range(200):
        opt2.zero_grad()
        loss2 = nn.BCELoss()(model(X_full).squeeze(), y_full)
        loss2.backward()
        opt2.step()

    return model
