import pickle
import os


def probability_tensor_to_classification_list(tensor, classification_threshold=1.2):
    classification_list = []

    classification_probs = tensor.tolist()
    for values in classification_probs:
        max_index = values.index(max(values))
        max_value = values[max_index]
        sorted_values_with_indices = sorted([(val, idx) for idx, val in enumerate(values)], reverse=True)
        second_max_value, second_max_index = sorted_values_with_indices[1]
        if max_value / second_max_value <= classification_threshold:
            classification_list.append([(max_index, max_value), (second_max_index, second_max_value)])
        else:
            classification_list.append([(max_index, max_value)])
    
    return classification_list


def handle_overlapping_edges(overlapping_edges, classification_list):
    for edge in overlapping_edges:
        index1, index2 = edge
        list1 = classification_list[index1]
        list2 = classification_list[index2]

        matching_indices = {x[0] for x in list1} & {x[0] for x in list2}

        if matching_indices:
            for match in matching_indices:
                if match == 2: continue

                if len(list1) == 1 and len(list2) == 1:
                    if list1[0][1] < list2[0][1]:
                        classification_list[index1] = [(1 - list1[0][0], list1[0][1])]
                    else:
                        classification_list[index2] = [(1 - list2[0][0], list2[0][1])]
                elif len(list1) == 1 and len(list2) == 2:
                    classification_list[index2] = [x for x in list2 if x[0] != match]
                elif len(list1) == 2 and len(list2) == 1:
                    classification_list[index1] = [x for x in list1 if x[0] != match]
                # CODE HERE: what if both have length 2?
    return classification_list


def infer_voices_from_raw(raw_pickle_dir, overlapping_pickle_dir, classification_threshold=1.2):
    for filename in os.listdir(raw_pickle_dir):
        if filename.endswith('.pkl'):
            raw_file_path = os.path.join(raw_pickle_dir, filename)
            overlapping_file_path = os.path.join(overlapping_pickle_dir, filename)

        with open(raw_file_path, 'rb') as f:
            tensor = pickle.load(f)

        classification_list = probability_tensor_to_classification_list(tensor, classification_threshold)

        with open(overlapping_file_path, 'rb') as f:
            overlapping_edges = pickle.load(f)

        if overlapping_edges:
            classification_list = handle_overlapping_edges(overlapping_edges, classification_list)

        classification_list = [[item[0] for item in sublist] for sublist in classification_list]
        modified_file_path = os.path.join('inference', 'voice_classification_inferred', filename)
        os.makedirs(os.path.dirname(modified_file_path), exist_ok=True)
        with open(modified_file_path, 'wb') as f:
            pickle.dump(classification_list, f)


if __name__ == "__main__":
    raw_pickle_dir = 'inference/voice_classification_raw'
    overlapping_pickle_dir = 'inference/overlapping_edges'
    infer_voices_from_raw(raw_pickle_dir, overlapping_pickle_dir, classification_threshold=1.2)

