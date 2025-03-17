from pickle import load, dump
import numpy as np
import os

def open_pickle(filepath):
    with open(filepath, 'rb') as f:
        clusters = load(f)
    return clusters

def get_clusters_in_terms_of_original_notes(clusters):
    final_clusters = [clusters[0]]
    for i, cluster in enumerate(clusters):
        if i == 0: continue
        final_clusters.append(np.matmul(final_clusters[-1], cluster))
    return final_clusters

def save_pickle(filepath, clusters):
    with open(filepath, 'wb') as f:
        dump(clusters, f)

def pickle_all_clusters(parent_directory):
    for item in os.listdir(parent_directory):
        # if item[:6] != "WTC_II": continue
        try:
            # Construct the full path of the item
            item_path = os.path.join(parent_directory, item)
            # Check if the item is a directory
            if os.path.isdir(item_path) and item not in ["mxls", "xmls"]:
                # print(item)
                fp = f"{item}/{item}.pkl_cluster"
                clusters = open_pickle(fp)
                new_clusters = get_clusters_in_terms_of_original_notes(clusters)
                save_pickle(fp + "_original", new_clusters)
                # print_pickle(fp[:-4] + "pkl")
        except FileNotFoundError as e:
            print(e)
            continue
        except IndexError as e:
            print(item)

if __name__ == "__main__":
    clusters = open_pickle("C:\\Users\\88ste\\PycharmProjects\\forks\\SchenkerGNN\\schenkerian_clusters\\Primi_1\\Primi_1.pkl_cluster_original")
    print(clusters)
    # pickle_all_clusters("C:\\Users\\88ste\\PycharmProjects\\forks\\GNN\\schenkerian_clusters")
    # clusters = open_pickle("C:\\Users\\88ste\\PycharmProjects\\forks\\GNN\\schenkerian_clusters\\Primi_1\\Primi_1.pkl")
    # new_clusters = get_clusters_in_terms_of_original_notes(clusters)
    # save_pickle("C:\\Users\\88ste\\PycharmProjects\\forks\\GNN\\schenkerian_clusters\\Primi_1\\Primi_1_repickle.pkl", new_clusters)
