import os
import matplotlib.pyplot as plt


def init_folder(PROJECT_ROOT_DIR, IMG_FOLDER, FILE_FOLDER):
    img_dir = os.path.join(PROJECT_ROOT_DIR, IMG_FOLDER)
    os.makedirs(img_dir, exist_ok=True)

    file_dir = os.path.join(PROJECT_ROOT_DIR, FILE_FOLDER)
    os.makedirs(file_dir, exist_ok=True)


# save the figures
def save_fig(fig_id, tight_layout=True):
    init_folder()
    path = os.path.join(PROJECT_ROOT_DIR, IMG_FOLDER, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# save files
def save_img_file(file, file_id):
    init_folder()
    path = os.path.join(PROJECT_ROOT_DIR, IMG_FOLDER, file_id)
    with open(path, 'w') as f:
        f.write(file)


# save dataframe to csv file
def save_dataframe(df, file_id, description=None):
    init_folder()
    path = os.path.join(PROJECT_ROOT_DIR, FILE_FOLDER, file_id)
    with open(path, 'a') as f:
        f.write('\n{}\n'.format(description))
    df.to_csv(path, index=None, sep='\t', mode='a')


if __name__ == '__main__':
    PROJECT_ROOT_DIR = "."
    IMG_FOLDER = "imgs"
    FILE_FOLDER = 'files'
    init_folder(PROJECT_ROOT_DIR, IMG_FOLDER, FILE_FOLDER)
