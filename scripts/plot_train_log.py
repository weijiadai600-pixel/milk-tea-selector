# scripts/plot_train_log.py
import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt


def main():
    log_path = "results/brand_classifier/train_log.csv"
    out_dir = "results/brand_classifier"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"找不到训练日志: {log_path}")

    df = pd.read_csv(log_path)

    # ------- Figure 1: loss -------
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_png = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_png, dpi=200)
    plt.close()

    # ------- Figure 2: acc -------
    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_png = os.path.join(out_dir, "acc_curve.png")
    plt.savefig(acc_png, dpi=200)
    plt.close()

    print("Saved:")
    print(" -", loss_png)
    print(" -", acc_png)


if __name__ == "__main__":
    main()
