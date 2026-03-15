import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Configuration class managing settings via environment variables."""

    data_dir: Path = Path(os.getenv("DATA_DIR", str(
        Path(__file__).parent / "data" / "olist")))
    seed: int = int(os.getenv("SEED", "42"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "128"))
    epochs: int = int(os.getenv("EPOCHS", "80"))
    lr: float = float(os.getenv("LR", "5e-4"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "1e-5"))
    logreg_lr: float = float(os.getenv("LOGREG_LR", "0.05"))
    logreg_iters: int = int(os.getenv("LOGREG_ITERS", "3000"))
    logreg_threshold: float = float(os.getenv("LOGREG_THRESHOLD", "0.35"))
    grad_clip_norm: float = float(os.getenv("GRAD_CLIP_NORM", "1.0"))
    logreg_l2: float = float(os.getenv("LOGREG_L2", "1e-4"))
    logreg_grad_clip: float = float(os.getenv("LOGREG_GRAD_CLIP", "5.0"))


FEATURE_COLS: List[str] = [
    "n_items",
    "total_price",
    "total_freight",
    "freight_ratio",
    "avg_weight_g",
    "product_volume_cm3",
    "approval_delay_hours",
    "estimated_days",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "same_state",
    "product_category",
    "seller_state",
    "customer_state",
]


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Determines the appropriate PyTorch device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class ShippingDataset(Dataset):
    """PyTorch Dataset for shipping features and delivery targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            X: Feature matrix.
            y: Target values.
        """
        assert X.shape[0] == y.shape[0], "Features and labels must have the same number of rows."
        self.X: torch.Tensor = torch.tensor(X, dtype=torch.float32)
        self.y: torch.Tensor = torch.tensor(
            y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------------------------
# ML Models
# -----------------------------------------------------------------------------
class LogisticRegressionScratch:
    """A simple Logistic Regression implementation from scratch."""

    def __init__(
        self,
        lr: float = 0.1,
        n_iters: int = 1000,
        pos_weight: float = 1.0,
        l2_lambda: float = 0.0,
        grad_clip: float = 5.0,
    ) -> None:
        self.lr: float = lr
        self.n_iters: int = n_iters
        self.pos_weight: float = pos_weight
        self.l2_lambda: float = l2_lambda
        self.grad_clip: float = grad_clip
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.losses: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function safely."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -80.0, 80.0)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Trains the logistic regression model using gradient descent."""
        assert X.shape[0] == y.shape[0], "Features and target must have the same length."
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        m, n = X.shape
        self.weights = np.zeros(n, dtype=np.float64)
        self.bias = 0.0

        sample_w: np.ndarray = np.where(y == 1, self.pos_weight, 1.0)

        for i in range(self.n_iters):
            with np.errstate(all="ignore"):
                logits: np.ndarray = X @ self.weights + self.bias
            y_hat: np.ndarray = self._sigmoid(logits)

            err: np.ndarray = (y_hat - y) * sample_w

            with np.errstate(all="ignore"):
                dw: np.ndarray = (X.T @ err) / m + \
                    (self.l2_lambda / m) * self.weights
            db: float = float(np.sum(err) / m)

            dw_norm: float = float(np.linalg.norm(dw))
            if dw_norm > self.grad_clip:
                dw = dw * (self.grad_clip / dw_norm)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if (i + 1) % 500 == 0:
                loss_vec: np.ndarray = -(
                    self.pos_weight * y * np.log(y_hat + 1e-9)
                    + (1.0 - y) * np.log(1.0 - y_hat + 1e-9)
                )
                l2_term: float = (self.l2_lambda / (2 * m)) * \
                    float(np.sum(self.weights**2))
                loss_value: float = float(np.mean(loss_vec) + l2_term)
                self.losses.append(loss_value)
                logger.debug(
                    f"LogReg Iter {i + 1}/{self.n_iters} Loss: {loss_value:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probability of the positive class."""
        assert self.weights is not None, "Model must be fitted before predicting."
        with np.errstate(all="ignore"):
            logits: np.ndarray = X.astype(
                np.float64) @ self.weights + self.bias
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts class labels based on a probability threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)


class DeliveryModel(nn.Module):
    """Neural Network model for delivery time regression."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        assert n_features > 0, "Number of features must be strictly positive."
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the network."""
        return self.net(x)


# -----------------------------------------------------------------------------
# Data Processing
# -----------------------------------------------------------------------------
def load_and_merge(data_dir: Path) -> pd.DataFrame:
    """
    Loads raw CSVs from the Olist dataset and merges them into a unified view.

    Args:
        data_dir: Path to the directory containing dataset CSVs.

    Returns:
        pd.DataFrame: Merged pandas dataframe containing orders, items, and products.
    """
    logger.info(f"Loading data from {data_dir}")
    assert data_dir.exists(), f"Directory not found: {data_dir}"

    try:
        orders = pd.read_csv(data_dir / "olist_orders_dataset.csv")
        items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")
        products = pd.read_csv(data_dir / "olist_products_dataset.csv")
        sellers = pd.read_csv(data_dir / "olist_sellers_dataset.csv")
        customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
        translation = pd.read_csv(
            data_dir / "product_category_name_translation.csv")
    except Exception as e:
        logger.error(f"Failed to load dataset files: {e}")
        raise

    orders = orders[orders["order_status"] == "delivered"].copy()

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    orders = orders.dropna(subset=["order_delivered_customer_date"])

    items = items.merge(
        products[
            [
                "product_id",
                "product_category_name",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]
        ],
        on="product_id",
        how="left",
    )
    items = items.merge(
        sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
    items = items.merge(translation, on="product_category_name", how="left")

    items_agg = items.groupby("order_id", as_index=False).agg(
        n_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_weight_g=("product_weight_g", "mean"),
        avg_length_cm=("product_length_cm", "mean"),
        avg_height_cm=("product_height_cm", "mean"),
        avg_width_cm=("product_width_cm", "mean"),
        product_category=("product_category_name_english", "first"),
        seller_state=("seller_state", "first"),
    )

    merged = orders.merge(
        customers[["customer_id", "customer_state"]], on="customer_id", how="left"
    ).merge(items_agg, on="order_id", how="left")

    assert not merged.empty, "Merged dataframe is empty after joins."
    logger.info(f"Successfully merged dataset. Shape: {merged.shape}")
    return merged


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Cleans, imputes, and engineers features for the shipping dataset.

    Args:
        df: Merged raw dataframe.

    Returns:
        Tuple containing the processed dataframe and a dictionary of fitted LabelEncoders.
    """
    logger.info("Starting preprocessing...")
    out = df.copy()

    # Time-based features
    out["actual_delivery_days"] = (
        out["order_delivered_customer_date"] - out["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    out["estimated_days"] = (
        out["order_estimated_delivery_date"] - out["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    out["is_late"] = (out["actual_delivery_days"] >
                      out["estimated_days"]).astype(int)
    out["approval_delay_hours"] = (
        out["order_approved_at"] - out["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600.0

    out["purchase_month"] = out["order_purchase_timestamp"].dt.month
    out["purchase_dayofweek"] = out["order_purchase_timestamp"].dt.dayofweek
    out["purchase_hour"] = out["order_purchase_timestamp"].dt.hour
    out["same_state"] = (out["seller_state"] ==
                         out["customer_state"]).astype(int)

    out["product_volume_cm3"] = out["avg_length_cm"] * \
        out["avg_height_cm"] * out["avg_width_cm"]
    out["freight_ratio"] = out["total_freight"] / (out["total_price"] + 1e-6)

    # Impute numeric columns
    numeric_cols = [
        "avg_weight_g",
        "avg_length_cm",
        "avg_height_cm",
        "avg_width_cm",
        "product_volume_cm3",
        "approval_delay_hours",
        "total_price",
        "total_freight",
        "freight_ratio",
        "n_items",
        "estimated_days",
    ]
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median())

    # Encode categorical columns
    cat_cols = ["product_category", "seller_state", "customer_state"]
    out[cat_cols] = out[cat_cols].fillna("unknown")

    encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].astype(str))
        encoders[col] = le

    out = (
        out.dropna(subset=["actual_delivery_days"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=FEATURE_COLS + ["actual_delivery_days", "is_late"])
    )

    assert not out.empty, "Dataframe is empty after dropping NAs."
    logger.info(
        f"Preprocessing complete. Rows: {len(out)}, Late Rate: {out['is_late'].mean():.3f}")
    return out, encoders


def split_data(
    df: pd.DataFrame, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Splits dataset into training and testing sets, and applies standard scaling."""
    logger.info("Splitting and scaling data...")
    X_raw: np.ndarray = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_reg: np.ndarray = df["actual_delivery_days"].to_numpy(dtype=np.float32)
    y_cls: np.ndarray = df["is_late"].to_numpy(dtype=np.float32)

    X_train_raw, X_test_raw, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X_raw, y_reg, y_cls, test_size=0.2, random_state=seed, stratify=y_cls
    )

    scaler = StandardScaler()
    X_train_sc: np.ndarray = scaler.fit_transform(X_train_raw)
    X_test_sc: np.ndarray = scaler.transform(X_test_raw)

    # Clip standardized features to prevent extreme outliers from causing matmul overflow
    X_train_sc = np.clip(X_train_sc, -5.0, 5.0)
    X_test_sc = np.clip(X_test_sc, -5.0, 5.0)

    assert X_train_sc.shape[0] == y_reg_train.shape[0], "Train split mismatch."
    assert X_test_sc.shape[0] == y_reg_test.shape[0], "Test split mismatch."

    logger.info(
        f"Split complete. Train Shape: {X_train_sc.shape}, Test Shape: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_reg_train, y_reg_test, y_cls_train, y_cls_test, scaler


# -----------------------------------------------------------------------------
# Training Pipeline
# -----------------------------------------------------------------------------
def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    y_mean: float,
    y_std: float,
) -> List[Dict[str, float]]:
    """
    Trains the PyTorch Delivery regression model.

    Returns:
        List of dictionaries containing loss history per epoch.
    """
    logger.info(
        f"Starting PyTorch training for {epochs} epochs on {device}...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        train_loss: float = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            yb_norm = (yb - y_mean) / y_std

            optimizer.zero_grad()
            pred_norm = model(Xb)
            loss = criterion(pred_norm, yb_norm)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss: float = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                yb_norm = (yb - y_mean) / y_std
                pred_norm = model(Xb)
                val_loss += criterion(pred_norm, yb_norm).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1:>3}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

    return history


def main() -> None:
    """Main execution pipeline."""
    cfg = Config()
    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"Runtime initialized using device: {device}")

    try:
        df = load_and_merge(cfg.data_dir)
        df,  _ = preprocess(df)
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, _scaler = split_data(
            df, cfg.seed
        )
    except AssertionError as e:
        logger.error(f"Data validation failed: {e}")
        return
    except Exception as e:
        logger.error(f"Pipeline failed during data loading/preprocessing: {e}")
        return

    # 1. Train Custom Logistic Regression (Classification)
    logger.info("Training Logistic Regression (Scratch)...")
    pos_weight = float(
        (len(y_cls_train) - y_cls_train.sum()) / y_cls_train.sum())
    logreg = LogisticRegressionScratch(
        lr=cfg.logreg_lr,
        n_iters=cfg.logreg_iters,
        pos_weight=pos_weight,
        l2_lambda=cfg.logreg_l2,
        grad_clip=cfg.logreg_grad_clip,
    )
    logreg.fit(X_train, y_cls_train)
    y_prob_lr = logreg.predict_proba(X_test)
    y_pred_lr = logreg.predict(X_test, threshold=cfg.logreg_threshold)

    logger.info(
        f"[LogReg Scratch] ROC-AUC: {roc_auc_score(y_cls_test, y_prob_lr):.3f}")
    logger.info(
        f"\n{classification_report(y_cls_test, y_pred_lr, target_names=['On Time', 'Late'])}"
    )

    # 2. Train PyTorch Model (Regression)
    logger.info("Training PyTorch Neural Network...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_reg_train, test_size=0.15, random_state=cfg.seed
    )

    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    assert y_std > 0, "Standard deviation of target is zero."

    train_loader = DataLoader(ShippingDataset(
        X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ShippingDataset(
        X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    model = DeliveryModel(n_features=X_train.shape[1]).to(device)
    train_pytorch_model(
        model,
        train_loader,
        val_loader,
        device,
        cfg.epochs,
        cfg.lr,
        cfg.weight_decay,
        cfg.grad_clip_norm,
        y_mean,
        y_std,
    )

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_pred_reg = (model(X_test_t).squeeze(
            1).cpu().numpy() * y_std) + y_mean

    rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_pred_reg)))
    mae = float(mean_absolute_error(y_reg_test, y_pred_reg))

    logger.info(f"[DeliveryModel PyTorch] RMSE: {rmse:.3f}")
    logger.info(f"[DeliveryModel PyTorch] MAE: {mae:.3f}")


if __name__ == "__main__":
    main()
