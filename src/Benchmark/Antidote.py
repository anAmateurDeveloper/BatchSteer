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

def train_model(model, X_train, y_train, lr=1e-2, epochs=50, verbose=False):
    optimzer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    model.train()
    for ep in range(epochs):
        optimzer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimzer.step()
        if verbose and (ep % 20 == 0):
            print(f"inner ep {ep} loss {loss.item():.4f}")
    return model

def evaluate(model, X, y, sensitive):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_t).numpy() > 0.5
    overall = accuracy_score(y, preds)
    # group accuracies
    g0 = accuracy_score(y[sensitive == 0], preds[sensitive == 0]) if np.sum(sensitive==0)>0 else 0.0
    g1 = accuracy_score(y[sensitive == 1], preds[sensitive == 1]) if np.sum(sensitive==1)>0 else 0.0
    return overall, g0, g1

def generate_antidote_data(X_train, y_train, sensitive_train,
                           n_antidote=20, antidote_steps=200, inner_epochs=40,
                           model_lr=1e-2, antidote_lr=1e-2, fairness_weight=1.0):
    """
    Learn antidote features+labels to reduce group accuracy gap.
    This is a simple bi-level-ish approximated optimization:
      - We treat antidote samples as variables (features and logits for label).
      - For each outer step we: copy base model, train for a few inner_epochs on (train + current antidote),
        compute the fairness loss (difference in group accuracies), compute gradient wrt antidote variables
        by backprop through the inner training steps (we use a simplistic approach of training with autograd).
    NOTE: For efficiency and numerical stability this is a toy method.
    """
    device = torch.device("cpu")

    n_features = X_train.shape[1]
    # initialize antidote features: sample from train distribution + small noise
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    antidote_X = torch.tensor(np.random.normal(mu, std, size=(n_antidote, n_features)), dtype=torch.float32, requires_grad=True, device=device)

    # initialize antidote labels as logits we will push to 0 or 1 using a sigmoid
    antidote_logits = torch.tensor(np.random.randn(n_antidote), dtype=torch.float32, requires_grad=True, device=device)

    # base model initialization (random)
    base_model = SimpleNet(n_features).to(device)

    outer_opt = optim.Adam([antidote_X, antidote_logits], lr=antidote_lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    s_train = np.array(sensitive_train)

    for outer in range(antidote_steps):
        # 1) build a fresh model and train it on train + current antidote (inner training)
        model = SimpleNet(n_features).to(device)
        # create inner optimizer that will be inside autograd graph: requires higher-order grad support
        # We'll simulate inner training by performing gradient steps and letting autograd track them.
        # Convert antidote labels to probabilities via sigmoid
        antidote_y_prob = torch.sigmoid(antidote_logits)
        # combine data
        X_comb = torch.cat([X_train_t, antidote_X], dim=0)
        y_comb = torch.cat([y_train_t, antidote_y_prob], dim=0)

        # inner training loop (we keep it differentiable)
        inner_opt = optim.SGD(model.parameters(), lr=model_lr)
        loss_fn = nn.BCELoss()
        for ep in range(inner_epochs):
            inner_opt.zero_grad()
            preds = model(X_comb)
            loss = loss_fn(preds, y_comb)
            loss.backward(create_graph=True)  # create graph to allow backprop to antidote params
            inner_opt.step()

        # evaluate fairness metric on validation (we use training here for simplicity)
        model.eval()
        with torch.no_grad():
            preds = model(X_train_t).cpu().numpy() > 0.5
        # compute group accuracies (we compute them using tensors but avoid breaking graph)
        # For gradient, we need a differentiable surrogate. We'll use group BCE losses as proxies:
        # group loss for group 0 and group 1, then fairness = abs(loss0 - loss1)
        # build masks as float tensors (non-differentiable indexes are fine)
        mask0 = torch.tensor((s_train == 0).astype(float), dtype=torch.float32, device=device)
        mask1 = torch.tensor((s_train == 1).astype(float), dtype=torch.float32, device=device)
        # compute differentiable group losses (use model's predictions on X_train combined)
        # Recompute preds on training set without torch.no_grad to keep graph
        preds_train = model(X_train_t)
        loss0 = loss_fn(preds_train * mask0, y_train_t * mask0) if mask0.sum() > 0 else torch.tensor(0.0, device=device)
        loss1 = loss_fn(preds_train * mask1, y_train_t * mask1) if mask1.sum() > 0 else torch.tensor(0.0, device=device)
        fairness_loss = torch.abs(loss0 - loss1)

        # we also want to keep model accuracy good: add overall training loss on combined set
        preds_comb = model(X_comb)
        overall_loss = loss_fn(preds_comb, y_comb)

        total_loss = overall_loss + fairness_weight * fairness_loss

        # Backpropagate through inner training to antidote variables
        outer_opt.zero_grad()
        total_loss.backward()
        # clip gradients to avoid exploding
        torch.nn.utils.clip_grad_norm_([antidote_X, antidote_logits], max_norm=5.0)
        outer_opt.step()

        if outer % 20 == 0:
            with torch.no_grad():
                overall, g0, g1 = evaluate(model, X_train, y_train, s_train)
                gap = abs(g0 - g1)
            print(f"[outer {outer}] total_loss {total_loss.item():.4f} train_acc {overall:.3f} g0 {g0:.3f} g1 {g1:.3f} gap {gap:.3f}")

    # return antidote data (features and hard labels)
    antidote_X_np = antidote_X.detach().cpu().numpy()
    antidote_labels_np = (torch.sigmoid(antidote_logits).detach().cpu().numpy() > 0.5).astype(int)
    return antidote_X_np, antidote_labels_np

if __name__ == "__main__":
    X, y = make_classification(n_samples=2000, n_features=8, n_informative=6, n_redundant=2, random_state=0)
    sensitive = (X[:, 0] > 0).astype(int)  # simple split by one feature

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, sensitive, test_size=0.3, random_state=0)

    # baseline model
    base_model = SimpleNet(X_train.shape[1])
    base_model = train_model(base_model, X_train, y_train, epochs=200, lr=1e-2)
    overall, g0, g1 = evaluate(base_model, X_test, y_test, s_test)
    print("Baseline test acc:", overall, "g0:", g0, "g1:", g1, "gap:", abs(g0-g1))

    antidote_X, antidote_y = generate_antidote_data(X_train, y_train, s_train,
                                                    n_antidote=50,
                                                    antidote_steps=200,
                                                    inner_epochs=40,
                                                    model_lr=1e-2,
                                                    antidote_lr=5e-3,
                                                    fairness_weight=5.0)

    # retrain final model with learned antidote appended
    X_aug = np.vstack([X_train, antidote_X])
    y_aug = np.concatenate([y_train, antidote_y])
    final_model = SimpleNet(X_train.shape[1])
    final_model = train_model(final_model, X_aug, y_aug, epochs=300, lr=1e-2)
    overall_f, g0_f, g1_f = evaluate(final_model, X_test, y_test, s_test)
    print("After antidote test acc:", overall_f, "g0:", g0_f, "g1:", g1_f, "gap:", abs(g0_f-g1_f))
