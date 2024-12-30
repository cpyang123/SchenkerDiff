import os
import glob


def delete_npy_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                os.remove(file_path)


if __name__ == "__main__":
    # Specify the directory path where you want to delete .npy files
    directory_path = "C:\\Users\\88ste\\PycharmProjects\\forks\\SchenkerGNN\\schenkerian_clusters"

    # Call the function to delete .npy files recursively
    delete_npy_files(directory_path)

