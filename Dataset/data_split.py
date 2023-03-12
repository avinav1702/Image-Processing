import splitfolders

input_folder = "D:\\SM\\Career_Projects\\Project_1_(Image Processing)\\Dataset\\Plant_leave_diseases_dataset_with_augmentation"
output_folder = "D:\\SM\\Career_Projects\\Project_1_(Image Processing)\\Dataset\\Augmented"

splitfolders.ratio (input_folder, output = output_folder, seed = 42, ratio = (.7, .2, .1), group_prefix = None)
