from json_to_cluster_refactor import get_clusters
import pickle
import os

def pickle_clusters(filepath):
    if filepath[-4:] != "json":
        raise ValueError("requires json file")
    clusters = get_clusters(filepath)
    new_fp = filepath[:-4] + "pkl_cluster"
    with open(new_fp, 'wb') as f:
        pickle.dump(clusters, f)


def print_pickle(filepath):
    with open(filepath, 'rb') as f:
        clusters = pickle.load(f)
        for cluster in clusters:
            print(cluster)

def pickle_all_clusters(parent_directory):
    for item in os.listdir(parent_directory):
        # if item[:6] != "WTC_II": continue
        try:
            # Construct the full path of the item
            item_path = os.path.join(parent_directory, item)
            # Check if the item is a directory
            if os.path.isdir(item_path) and item not in ["mxls", "xmls"]:
                # print(item)
                fp = f"{item}/{item}.json"
                pickle_clusters(fp)
                # print_pickle(fp[:-4] + "pkl")
        except FileNotFoundError as e:
            print(e)
            continue
        except IndexError as e:
            print(item)

if __name__ == "__main__":
    # fp = "WTC_II_B_maj/WTC_II_B_maj"
    # pickle_clusters(fp + ".json")
    # print_pickle(fp + ".pkl")
    parent_directory = "."
    pickle_all_clusters(parent_directory)

