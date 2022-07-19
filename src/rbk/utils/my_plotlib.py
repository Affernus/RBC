import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_word_cloud(
    clusters_word_freq: dict,
    single_image_ratio: float=2.0,
    size_coef: int=5,
    n_cols: int=3
) -> None:
    """Визуализирует частотности слов в кластерах

    Args:
        clusters_word_freq (dict): словарь с частотностями слов в кластерах, ключи - кластеры
        single_image_ratio (float, optional): _description_. Defaults to 2.0.
        size_coef (int, optional): _description_. Defaults to 5.
        n_cols (int, optional): _description_. Defaults to 3.
    """

    n_clusters = len(clusters_word_freq)
    n_rows = int( np.ceil(n_clusters / n_cols) )

    width = size_coef * n_cols
    height = n_rows * width / (n_cols*single_image_ratio)

    _, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(width, height))
    axs_flat = axs.flatten()
    for i, ax in enumerate(axs_flat):
        ax.axis('off')
        if i < n_clusters:
            cluster_size = len(clusters_word_freq[i])
            if cluster_size == 0:
                continue
            cloud =  WordCloud().generate_from_frequencies(clusters_word_freq[i])
            ax.imshow(cloud, interpolation='bilinear')
            ax.set_title(f'cluster #{i}, {cluster_size} unique words')

    plt.tight_layout()
    plt.show()
