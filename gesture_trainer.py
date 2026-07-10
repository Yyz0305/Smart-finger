"""
手势模型训练器 — 支持 Random Forest + PyTorch MLP
包含交叉验证、混淆矩阵、干扰分析
"""

import os
import json
import glob
import numpy as np
from collections import Counter

import joblib

# ─── scikit-learn ───
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# ─── PyTorch ───
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "gesture_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "gesture_models")
os.makedirs(MODEL_DIR, exist_ok=True)

GESTURE_CLASSES = [
    "five_pinch", "three_down", "open_palm", "fist", "point_up",
    "ok", "victory", "thumb_up", "thumb_down", "rest",
]
CLASS_TO_ID = {n: i for i, n in enumerate(GESTURE_CLASSES)}
ID_TO_CLASS = {i: n for i, n in enumerate(GESTURE_CLASSES)}


# ─── 数据加载 ────────────────────────────────────────────────

def load_all_data(data_dir=DATA_DIR):
    """加载所有JSON数据文件"""
    files = glob.glob(os.path.join(data_dir, "gesture_data_*.json"))
    if not files:
        print(f"未找到数据文件: {data_dir}/gesture_data_*.json")
        return None, None, None

    X, y, meta = [], [], []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        for cls_name, samples in data.items():
            if cls_name not in CLASS_TO_ID:
                print(f"  跳过未知类别: {cls_name}")
                continue
            label = CLASS_TO_ID[cls_name]
            for s in samples:
                X.append(s["features"])
                y.append(label)
                meta.append({"file": os.path.basename(fpath), "class": cls_name})

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"  加载 {len(X)} 样本, {len(np.unique(y))} 类别")
    print(f"  类别分布: {dict(Counter(y))}")
    return X, y, meta


# ─── 数据增强 ────────────────────────────────────────────────

def augment_data(X, y, noise_std=0.003, flip_p=0.3):
    """数据增强: 加噪声 + 水平翻转"""
    X_aug, y_aug = [X], [y]

    # 高斯噪声
    X_noise = X + np.random.normal(0, noise_std, X.shape)
    X_aug.append(X_noise)
    y_aug.append(y)

    # 水平翻转 (handedness swap for x coordinates)
    # x坐标每3个一组 (x,y,z), 翻转x: 1-x
    X_flip = X.copy()
    X_flip[:, 0::3] = 1.0 - X[:, 0::3]
    X_aug.append(X_flip)
    y_aug.append(y)

    X_all = np.vstack(X_aug)
    y_all = np.hstack(y_aug)
    print(f"  数据增强: {len(X)} → {len(X_all)} 样本")
    return X_all, y_all


# ─── Random Forest ──────────────────────────────────────────

def train_random_forest(X_train, y_train, X_test, y_test):
    """训练 Random Forest 分类器"""
    print("\n─── Random Forest ───")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=4,
        min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    print(f"  Test Accuracy: {acc:.4f}")

    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=GESTURE_CLASSES,
                                labels=range(len(GESTURE_CLASSES)), zero_division=0))
    return rf, acc


# ─── PyTorch MLP ─────────────────────────────────────────────

class GestureMLP(nn.Module):
    def __init__(self, input_dim=63, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, y_test, epochs=60, batch_size=32):
    """训练 PyTorch MLP"""
    print("\n─── PyTorch MLP ───")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Normalize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    model = GestureMLP(input_dim=X_train.shape[1], num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_s), torch.LongTensor(y_test)),
        batch_size=batch_size,
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {train_loss/len(train_loader):.4f}  Acc: {acc:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best Test Acc: {best_acc:.4f}")

    # 最终评估
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            preds = model(Xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    print(classification_report(all_true, all_preds, target_names=GESTURE_CLASSES,
                                labels=range(n_classes), zero_division=0))

    return model, scaler, best_acc


# ─── 交叉验证 ────────────────────────────────────────────────

def cross_validate_models(X, y, n_folds=5):
    """K折交叉验证比较RF和MLP"""
    print(f"\n{'='*50}")
    print(f"  {n_folds}-Fold 交叉验证")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    rf_scores = []
    mlp_scores = []
    fold = 1

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # RF
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        rf_scores.append(rf.score(X_te, y_te))

        # MLP (simplified for CV speed)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GestureMLP(input_dim=X.shape[1], num_classes=len(np.unique(y))).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 快速训练
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_s), torch.LongTensor(y_tr)),
                           batch_size=32, shuffle=True)
        for epoch in range(30):
            model.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(Xb), yb).backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_te_t = torch.FloatTensor(X_te_s).to(device)
            preds = model(X_te_t).argmax(1).cpu().numpy()
        mlp_scores.append((preds == y_te).mean())

        print(f"  Fold {fold}: RF={rf_scores[-1]:.4f}  MLP={mlp_scores[-1]:.4f}")
        fold += 1

    print(f"\n  RF  CV Accuracy: {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
    print(f"  MLP CV Accuracy: {np.mean(mlp_scores):.4f} ± {np.std(mlp_scores):.4f}")
    return rf_scores, mlp_scores


# ─── 混淆矩阵 ────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=GESTURE_CLASSES[:cm.shape[0]])
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    print(f"  混淆矩阵已保存: {path}")
    plt.close()


# ─── 干扰分析 ────────────────────────────────────────────────

def analyze_interference(model, X, y, model_type="rf", scaler=None):
    """分析混淆最多的类别对(干扰分析)"""
    if model_type == "rf":
        y_pred = model.predict(X)
    elif model_type == "mlp" and scaler is not None:
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_s = scaler.transform(X)
            y_pred = model(torch.FloatTensor(X_s).to(device)).argmax(1).cpu().numpy()

    cm = confusion_matrix(y, y_pred)
    n = cm.shape[0]

    # 找出最大的混淆对
    confusion_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                confusion_pairs.append((cm[i][j], i, j))

    confusion_pairs.sort(reverse=True)

    print(f"\n  ─── 干扰分析 (Top-5 易混淆类别对) ───")
    for count, true_idx, pred_idx in confusion_pairs[:5]:
        pct = count / cm[true_idx].sum() * 100
        print(f"    {ID_TO_CLASS[true_idx]:12s} → {ID_TO_CLASS[pred_idx]:12s}: "
              f"{count} 次 ({pct:.1f}%)")


# ─── 主流程 ──────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  手势模型训练器")
    print("=" * 50)

    # 1. 加载数据
    X, y, meta = load_all_data()
    if X is None:
        print("请先运行 gesture_data_collector.py 采集数据")
        return

    # 2. 数据增强
    X, y = augment_data(X, y)

    # 3. 分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    print(f"\n  训练: {len(X_train)}  测试: {len(X_test)}")

    # 4. 交叉验证
    rf_cv, mlp_cv = cross_validate_models(X, y)

    # 5. 训练 RF
    rf_model, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)

    # 6. 训练 MLP
    mlp_model, mlp_scaler, mlp_acc = train_mlp(X_train, y_train, X_test, y_test)

    # 7. 混淆矩阵
    y_pred_rf = rf_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_rf, "Random Forest - Confusion Matrix")

    # 8. 干扰分析
    analyze_interference(rf_model, X_test, y_test, "rf")
    analyze_interference(mlp_model, X_test, y_test, "mlp", mlp_scaler)

    # 9. 保存模型
    rf_path = os.path.join(MODEL_DIR, "gesture_rf.joblib")
    joblib.dump(rf_model, rf_path)
    print(f"\n  RF 模型已保存: {rf_path}")

    mlp_path = os.path.join(MODEL_DIR, "gesture_mlp.pth")
    torch.save({
        "model_state": mlp_model.state_dict(),
        "input_dim": X.shape[1],
        "num_classes": len(np.unique(y)),
        "scaler_mean": mlp_scaler.mean_.tolist(),
        "scaler_scale": mlp_scaler.scale_.tolist(),
    }, mlp_path)
    print(f"  MLP模型已保存: {mlp_path}")

    # 10. 类别映射
    mapping_path = os.path.join(MODEL_DIR, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({"id_to_class": ID_TO_CLASS, "class_to_id": CLASS_TO_ID}, f, indent=1)
    print(f"  类别映射已保存: {mapping_path}")

    print(f"\n{'='*50}")
    print(f"  RF  Accuracy: {rf_acc:.4f}")
    print(f"  MLP Accuracy: {mlp_acc:.4f}")
    print(f"  RF  CV: {np.mean(rf_cv):.4f} ± {np.std(rf_cv):.4f}")
    print(f"  MLP CV: {np.mean(mlp_cv):.4f} ± {np.std(mlp_cv):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
