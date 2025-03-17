import os

if __name__ == "__main__":
    # List of folder names
    # folder_names = [
    #     # "WTC_I_A_maj",
    #     "WTC_II_A_min",
    #     # "WTC_I_Ab_maj",
    #     "WTC_II_B_maj",
    #     "WTC_II_B_min",
    #     "WTC_II_Bb_maj",
    #     "WTC_II_Bb_min",
    #     "WTC_II_C#_maj",
    #     "WTC_II_C#_min",
    #     # "WTC_I_C#_min_subject2",
    #     "WTC_II_C_maj",
    #     "WTC_II_C_min",
    #     "WTC_II_D_maj",
    #     "WTC_II_D_min",
    #     "WTC_II_E_maj",
    #     "WTC_II_E_min",
    #     "WTC_II_Eb_maj",
    #     # "WTC_I_F#_maj",
    #     # "WTC_I_F#_min",
    #     "WTC_II_F_maj",
    #     # "WTC_I_F_min",
    #     # "WTC_I_g#_min",
    #     "WTC_II_G_maj",
    #     # "WTC_I_G_min"
    # ]
    num_pachelbels = [23, 10, 11, 8, 12, 10, 8, 13]
    names = ["Primi", "Secundi", "Tertii", "Quarti", "Quinti", "Sexti", "Septimi", "Octavi"]
    folder_names = [
        f"{name}_{num}" for name, max_num in zip(names, num_pachelbels) for num in range(1, max_num + 1)
    ]
    print(folder_names)

    # Parent directory where folders will be created
    # parent_directory = "output_folders"
    #
    # # Create parent directory if it doesn't exist
    # if not os.path.exists(parent_directory):
    #     os.makedirs(parent_directory)

    # Create folders with names from the list
    for folder_name in folder_names:
        # folder_path = os.path.join(parent_directory, folder_name)
        try:
            os.makedirs(folder_name)
            print("Created folder:", folder_name)
        except Exception as e:
            print(e)




