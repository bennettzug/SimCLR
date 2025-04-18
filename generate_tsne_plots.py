import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from openTSNE import TSNE
from sklearn.cluster import KMeans
from tqdm import tqdm


def plot_tsne(embeddings, labels, epoch, output_dir, perplexity=30, use_kmeans=False, n_clusters=10):
    tsne = TSNE(n_components=2, perplexity=perplexity)
    embeddings_2d = tsne.fit(embeddings)

    plt.figure(figsize=(10, 8))

    if use_kmeans or len(np.unique(labels)) <= 1:
        print(f"Using K-means clustering with {n_clusters} clusters (true labels not available)")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            plt.scatter(
                embeddings_2d[cluster_labels == label, 0],
                embeddings_2d[cluster_labels == label, 1],
                color=colors[i],
                label=f"Class {label}",
                alpha=0.7,
                s=20,
            )

        plt.title(f"t-SNE Visualization at Epoch {epoch}")
    else:
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            plt.scatter(
                embeddings_2d[labels == label, 0],
                embeddings_2d[labels == label, 1],
                color=colors[i],
                label=f"Class {label}",
                alpha=0.7,
                s=20,
            )

        plt.title(f"t-SNE Visualization at Epoch {epoch}")

    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"tsne_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved t-SNE plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate t-SNE plots for embeddings")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the log directory")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Path to the output directory (default: log_dir/tsne_plots)"
    )
    parser.add_argument("--perplexity", type=int, default=30, help="Perplexity for t-SNE (default: 30)")
    parser.add_argument("--use-kmeans", action="store_true", help="Use k-means clustering for t-SNE")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for k-means (default: 10)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.log_dir, "tsne_plots")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    embeddings_dir = os.path.join(args.log_dir, "embeddings")
    embedding_files = sorted(glob.glob(os.path.join(embeddings_dir, "embeddings_epoch_*.npz")))

    labeled_embeddings_dir = os.path.join(args.log_dir, "labeled_embeddings")
    labeled_embedding_files = sorted(glob.glob(os.path.join(labeled_embeddings_dir, "embeddings_epoch_*.npz")))

    if not labeled_embedding_files:
        print(f"No labeled embedding files found in {labeled_embeddings_dir}")
        print(f"Found {len(embedding_files)} embedding files in {embeddings_dir}")

        for file_path in tqdm(embedding_files, desc="Generating t-SNE plots"):
            epoch = int(os.path.basename(file_path).split("_")[-1].split(".")[0])

            data = np.load(file_path)
            embeddings = data["embeddings"]
            labels = data["labels"]

            plot_tsne(embeddings, labels, epoch, args.output_dir, perplexity=args.perplexity)
    else:
        print(f"Found {len(labeled_embedding_files)} labeled embedding files in {labeled_embeddings_dir}")

        for file_path in tqdm(labeled_embedding_files, desc="Generating t-SNE plots"):
            epoch = int(os.path.basename(file_path).split("_")[-1].split(".")[0])

            data = np.load(file_path)
            embeddings = data["embeddings"]
            labels = data["labels"]
            # eventually pull labels from the file in /datasets/datasetname

            plot_tsne(embeddings, labels, epoch, args.output_dir, perplexity=args.perplexity)


if __name__ == "__main__":
    main()
