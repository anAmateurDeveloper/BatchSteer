import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

torch.manual_seed(0)
np.random.seed(0)


class SimpleNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.lin(x)).squeeze(-1)


def train_model(model, X_train, y_train, lr=1e-2, epochs=100, verbose=False):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        if verbose and (ep % 50 == 0):
            print(f"[train] ep {ep} loss {loss.item():.4f}")
    return model


def evaluate(model, X, y, sensitive):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = (model(X_t).cpu().numpy() > 0.5).astype(int)
    overall = accuracy_score(y, preds)
    g0 = accuracy_score(y[sensitive == 0], preds[sensitive == 0]) if np.sum(sensitive == 0) > 0 else 0.0
    g1 = accuracy_score(y[sensitive == 1], preds[sensitive == 1]) if np.sum(sensitive == 1) > 0 else 0.0
    return overall, g0, g1


def _closest_points_mean(X_group):
    """Return the mean vector of X_group. Simple prototype for targeting."""
    return X_group.mean(axis=0)


def cast_generate(X_train, y_train, sensitive_train,
                  n_antidote=40,
                  antidote_steps=300,
                  inner_epochs=30,
                  model_lr=1e-2,
                  antidote_lr=5e-3,
                  fairness_weight=5.0,
                  realism_weight=0.1,
                  clip_to_data_range=True,
                  verbose_every=50,
                  device="cpu"):
    """
    CAST-style corrective antidote generator.

    Args:
      X_train: numpy array (N, d)
      y_train: numpy array (N,) binary labels {0,1}
      sensitive_train: numpy array (N,) binary sensitive attribute {0,1}
      n_antidote: total number of synthetic points to learn
      antidote_steps: number of outer optimization steps
      inner_epochs: inner training epochs per outer step (differentiable)
      model_lr: learning rate for inner model training
      antidote_lr: learning rate for antidote parameter optimizer
      fairness_weight: how strongly to optimize fairness (higher -> more fairness emphasis)
      realism_weight: regularizer weight keeping antidote near data prototypes
      clip_to_data_range: if True, clip learned features to observed min/max per feature
      device: "cpu" or "cuda"

    Returns:
      antidote_X_np: (n_antidote, d)
      antidote_labels_np: (n_antidote,) ints {0,1}
    """
    device = torch.device(device)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).astype(int)
    s_train = np.asarray(sensitive_train).astype(int)

    N, d = X_train.shape

    # Compute per-group prototypes to initialize antidote toward disadvantaged group distribution
    X_g0 = X_train[s_train == 0]
    X_g1 = X_train[s_train == 1]

    proto0 = _closest_points_mean(X_g0) if len(X_g0) > 0 else X_train.mean(axis=0)
    proto1 = _closest_points_mean(X_g1) if len(X_g1) > 0 else X_train.mean(axis=0)

    # initialize antidote features by sampling near both prototypes.
    # We'll allocate antidote budget adaptively later; initialize evenly.
    n0 = n_antidote // 2
    n1 = n_antidote - n0

    # small gaussian noise around prototypes
    antidote_init_0 = np.random.normal(loc=proto0, scale=0.1 * (X_train.std(axis=0) + 1e-6), size=(n0, d))
    antidote_init_1 = np.random.normal(loc=proto1, scale=0.1 * (X_train.std(axis=0) + 1e-6), size=(n1, d))
    antidote_X_np = np.vstack([antidote_init_0, antidote_init_1])

    # make features torch params
    antidote_X = torch.tensor(antidote_X_np, dtype=torch.float32, requires_grad=True, device=device)

    # labels as logits
    antidote_logits = torch.tensor(np.random.randn(n_antidote), dtype=torch.float32, requires_grad=True, device=device)

    # data tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    # data feature ranges for clipping realism
    if clip_to_data_range:
        data_min = X_train.min(axis=0)
        data_max = X_train.max(axis=0)
        data_min_t = torch.tensor(data_min, dtype=torch.float32, device=device)
        data_max_t = torch.tensor(data_max, dtype=torch.float32, device=device)
    else:
        data_min_t = None
        data_max_t = None

    outer_opt = optim.Adam([antidote_X, antidote_logits], lr=antidote_lr)

    loss_fn = nn.BCELoss(reduction="mean")

    for outer in range(antidote_steps):
        # 1) create a fresh model for inner training (so gradients can flow to antidote params)
        model = SimpleNet(d).to(device)

        # determine which group is currently disadvantaged by training a quick baseline model on train only
        # (we can also evaluate using running model but using a tiny quick pass is fine)
        # We'll compute group losses after inner training; but we want to bias initial allocation:
        # compute current group losses with untrained model (random) is pointless; instead we'll use a small trained baseline only once before loop.
        # For simplicity, we'll do dynamic allocation based on current model after inner training below.

        # prepare combined dataset (train + current antidote)
        antidote_y_prob = torch.sigmoid(antidote_logits)
        X_comb = torch.cat([X_train_t, antidote_X], dim=0)
        y_comb = torch.cat([y_train_t, antidote_y_prob], dim=0)

        # inner training (differentiable): do small number of gradient steps while retaining graph
        inner_opt = optim.SGD(model.parameters(), lr=model_lr)
        for ep in range(inner_epochs):
            inner_opt.zero_grad()
            preds = model(X_comb)
            loss_inner = loss_fn(preds, y_comb)
            # inner step with create_graph=True on backward to allow gradients to flow to antidote params
            loss_inner.backward(create_graph=True)
            inner_opt.step()

        # After inner training, compute group-specific losses (differentiable) on original training set
        model.eval()
        preds_train = model(X_train_t)  # keep requires_grad to compute fairness gradient
        mask0 = torch.tensor((s_train == 0).astype(float), dtype=torch.float32, device=device)
        mask1 = torch.tensor((s_train == 1).astype(float), dtype=torch.float32, device=device)

        # compute group BCE losses using masked values; to avoid dividing by zero, guard with max(1, sum)
        if mask0.sum() > 0:
            loss0 = (loss_fn(preds_train * mask0, y_train_t * mask0) * (mask0.numel() / (mask0.sum() + 1e-8)))
        else:
            loss0 = torch.tensor(0.0, device=device)
        if mask1.sum() > 0:
            loss1 = (loss_fn(preds_train * mask1, y_train_t * mask1) * (mask1.numel() / (mask1.sum() + 1e-8)))
        else:
            loss1 = torch.tensor(0.0, device=device)

        # fairness objective: reduce absolute difference of group losses
        fairness_loss = torch.abs(loss0 - loss1)

        # overall combined loss to ensure model still learns well
        preds_comb = model(X_comb)
        overall_loss = loss_fn(preds_comb, y_comb)

        # realism regularizer: push antidote features toward nearest group prototype (encourages realistic features)
        # find which group currently has higher loss: disadvantaged group
        disadvantaged_group = 0 if (loss0.item() > loss1.item()) else 1
        prototype = proto0 if disadvantaged_group == 0 else proto1
        prototype_t = torch.tensor(prototype, dtype=torch.float32, device=device)
        # We only regularize antidote examples allocated to the disadvantaged group
        # Allocate indices: simple heuristic — first half to g0, second half to g1 as initialized
        n0 = n_antidote // 2
        n1 = n_antidote - n0
        if disadvantaged_group == 0:
            targeted_idx = torch.arange(0, n0, device=device)
        else:
            targeted_idx = torch.arange(n0, n0 + n1, device=device)
        if len(targeted_idx) > 0:
            targeted_X = antidote_X[targeted_idx]
            realism_reg = ((targeted_X - prototype_t) ** 2).mean()
        else:
            realism_reg = torch.tensor(0.0, device=device)

        total_loss = overall_loss + fairness_weight * fairness_loss + realism_weight * realism_reg

        # Backpropagate to antidote parameters
        outer_opt.zero_grad()
        total_loss.backward()
        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([antidote_X, antidote_logits], max_norm=5.0)
        outer_opt.step()

        # clip antidote features to data range for realism if requested
        if clip_to_data_range:
            with torch.no_grad():
                antidote_X.data = torch.max(torch.min(antidote_X.data, data_max_t), data_min_t)

        # OPTIONAL: re-balance number of antidote samples allocated to disadvantaged group adaptively
        # Here we do a simple heuristic: if gap is large, shift more antidote capacity to disadvantaged group.
        # (We won't reshape arrays — we keep fixed number but we can nudge prototypes)
        if outer % verbose_every == 0 or outer == antidote_steps - 1:
            # Evaluate current model on training or a small validation set
            with torch.no_grad():
                overall_acc, g0_acc, g1_acc = evaluate(model, X_train, y_train, s_train)
                gap = abs(g0_acc - g1_acc)
            print(f"[outer {outer}] total_loss {total_loss.item():.4f} train_acc {overall_acc:.3f} g0 {g0_acc:.3f} g1 {g1_acc:.3f} gap {gap:.3f}")

    # finalize antidote dataset: features and hard labels
    with torch.no_grad():
        final_X = antidote_X.detach().cpu().numpy()
        final_labels = (torch.sigmoid(antidote_logits).detach().cpu().numpy() > 0.5).astype(int)

    return final_X, final_labels


if __name__ == "__main__":
    X, y = make_classification(n_samples=2000, n_features=8, n_informative=6, n_redundant=2, random_state=0)
    sensitive = (X[:, 0] > 0).astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, sensitive, test_size=0.3, random_state=0)

    # Baseline
    base_model = SimpleNet(X_train.shape[1])
    base_model = train_model(base_model, X_train, y_train, epochs=250, lr=1e-2)
    overall, g0, g1 = evaluate(base_model, X_test, y_test, s_test)
    print("Baseline test acc:", overall, "g0:", g0, "g1:", g1, "gap:", abs(g0 - g1))

    # Generate CAST antidote
    antidote_X, antidote_y = cast_generate(X_train, y_train, s_train,
                                           n_antidote=80,
                                           antidote_steps=300,
                                           inner_epochs=30,
                                           model_lr=1e-2,
                                           antidote_lr=1e-2,
                                           fairness_weight=8.0,
                                           realism_weight=0.05,
                                           clip_to_data_range=True,
                                           verbose_every=50,
                                           device="cpu")

    # Retrain from scratch on train + learned antidote
    X_aug = np.vstack([X_train, antidote_X])
    y_aug = np.concatenate([y_train, antidote_y])
    final_model = SimpleNet(X_train.shape[1])
    final_model = train_model(final_model, X_aug, y_aug, epochs=300, lr=1e-2)
    overall_f, g0_f, g1_f = evaluate(final_model, X_test, y_test, s_test)
    print("After CAST test acc:", overall_f, "g0:", g0_f, "g1:", g1_f, "gap:", abs(g0_f - g1_f))
