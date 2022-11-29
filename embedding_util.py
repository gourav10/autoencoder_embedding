import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


def get_latent_space_embeddings(data, device, encoder):

    encoded_samples = []
    for sample in tqdm(data):
        img = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        # Encode image
        encoder.eval()
        with torch.no_grad():
            encoded_img = encoder(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {
            f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)
        }
        encoded_sample["label"] = label
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)
    return encoded_samples


def visualize_embeddings_2D(x_values, y_values, labels):
    plt.figure(figsize=(17, 9))
    plt.scatter(x_values, y_values, c=labels, cmap="tab10")
    plt.colorbar()
    plt.show()


def cluster_latent_space(clustering_model, embedding_df):
    clustering_results = clustering_model.fit_transform(
        embedding_df.drop(["label"], axis=1)
    )
    visualize_embeddings_2D(
        clustering_results[:, 0], clustering_results[:, 1], embedding_df.label
    )
