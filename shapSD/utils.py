import os
import matplotlib.pyplot as plt

PROJECT_ROOT_DIR = ".."
IMG_FOLDER = "imgs"

# save the figures
def save_fig(fig_id, tight_layout=True):

    path = os.path.join(PROJECT_ROOT_DIR, IMG_FOLDER, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# save files
def save_file(file, file_id):
    path = os.path.join(PROJECT_ROOT_DIR, IMG_FOLDER, file_id)
    with open(path, 'w') as f:
        f.write(file)
