from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    input_path = Path("reports") / "hidden_size_results.csv"
    output_path = Path("reports") / "hidden_size_heatmap.png"

    df = pd.read_csv(input_path)

    df["validation_accuracy_percent"] = df["final_valid_accuracy"] * 100

    heatmap_data = df.pivot(
        index="units2",
        columns="units1",
        values="validation_accuracy_percent",
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(heatmap_data, cmap="RdYlGn_r")

    ax.set_title("Validation Accuracy by Hidden-Layer Size")
    ax.set_xlabel("units1")
    ax.set_ylabel("units2")

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)

    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    for row_index, units2 in enumerate(heatmap_data.index):
        for col_index, units1 in enumerate(heatmap_data.columns):
            value = heatmap_data.loc[units2, units1]
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}%",
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax, label="Validation accuracy (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved heatmap to: {output_path}")


if __name__ == "__main__":
    main()