import numpy as np
import os
from scipy import misc
import time

# Paths
ABS_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = ABS_PATH + '\\MODELS\\'
TO_PROCESS_PATH = ABS_PATH + '\\TO_PROCESS\\'
RESULTS_PATH = ABS_PATH + '\\RESULTS\\'

# Frame dimensions
# All the images will be cropped to DIM x DIM
DIM = 55


def folders_in_path(path):
    """
    Get the names of all the folders in path.
    :param path (string) - target path.
    :return: folders_list (list of string) - list of the folder names.
    """
    folders_list = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            folders_list.append(name)
    return folders_list


def load_imgs_from_folder(path):
    """
    Load all the glyph images in path, crop them, and put
    them into a numpy array.
    :param path (string) - target path.
    :return: data_ndarr (ndarray) shape = (num_images, DIM, DIM)
             data_ndarr[i,:,:] corresponds to the i-th
             image in path. 
    """

    files_names = [f_name for f_name in os.listdir(path)
                   if os.path.isfile(os.path.join(path, f_name))]

    # Extract file name whithout extension
    files_names = [f_name[:-4] for f_name in files_names]

    # Allocate ndarray (all processed images will have the same size)
    data_ndarr = np.zeros((len(files_names), DIM, DIM))

    for i, f_name in enumerate(files_names):
        im = misc.imread(path + f_name + '.png', mode='L')
        # Threshold to obtain all elements of either 0 or 1
        im[im > 1] = 0
        # Invert 0's to 1's
        im = 1 - im
        # To avoid the limit case where the image is only black pixels
        if np.count_nonzero(im) > 0:
            # Do bounding box around image
            im = bbox(im)
            # Place the cropped image in the corner of slice i of data_ndarr
            data_ndarr[i, :im.shape[0], :im.shape[1]] = im

    return files_names, data_ndarr.astype(int)


def bbox(img):
    """
    Crop an image around its bounding box.
    :param img (ndarray) containing a black and white image
           of the glyph.
    :return: cropped img (ndarray).
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return img[rmin:rmax, cmin:cmax]


def get_match(proc_img_ndarr, model_ndarr, n_model):
    """
    For all the model glyph images, get the best match to 
    the glyph contained in proc_img_ndarr.
    :param proc_img_ndarr (ndarray) - the glyph image we want to find
           the match for.
    :param model_ndarr (ndarray) - contains all the model glyph images.
    :param n_model (int) number of model images.
    :return: the index (int) of the most similar model glyph image to
             proc_img_ndarr.
    """
    _sum_abs_diffs = np.zeros(n_model)
    for m in range(n_model):
        # Absolute difference (ndarray)
        _diff_arr = np.abs(proc_img_ndarr - model_ndarr[m, :, :])
        _sum_abs_diffs[m] = np.sum(_diff_arr)
    # The lower the value of the summed abs difference, the more similar
    match_idx = np.argmin(_sum_abs_diffs)
    return match_idx.astype(int)


if __name__ == '__main__':
    print('\nLoading model data...')
    model_files_names, model_ndarr = load_imgs_from_folder(MODEL_PATH)
    n_model = model_ndarr.shape[0]
    print('Done. Found ' + str(n_model) + ' model images')

    print('\nLooking for folders to process...')
    # Get folder names under to_process_path
    _folders_list = folders_in_path(TO_PROCESS_PATH)
    print('Done. Found ' + str(len(_folders_list)) + ' folders to process, with names:')
    print(_folders_list)

    print('\nWill now iterate throughout the folders to process, one by one.')
    print('------------------')

    # Iterate through all the folders, and find the matches
    for i, f in enumerate(_folders_list):
        start_time = time.time()
        print('\nLoading data in folder ' + f + '...')
        proc_files_names, proc_ndarr = load_imgs_from_folder(TO_PROCESS_PATH + '\\' + f + '\\')
        n_process = proc_ndarr.shape[0]
        print('Done. Found ' + str(n_process) + ' images to process.')
        print('Computing matches...')
        sum_abs_diff = np.zeros((n_model, n_process))

        # Multi-threading. Doesn't yield faster results than non-multi-threading.
        # match_idxs = Parallel(n_jobs=4)(delayed(get_match)(proc_ndarr[p, :, :], model_ndarr, n_model)
        #                                             for p in range(n_process))

        match_idxs = np.zeros(n_process).astype(int)
        for p in range(n_process):
            match_idxs[p] = get_match(proc_ndarr[p, :, :], model_ndarr, n_model)

        # Write to .csv
        print('Done.')
        csv_file_name = RESULTS_PATH + f + '.csv'
        print('Writing results to ' + csv_file_name + '...')
        with open(csv_file_name, 'w') as file:
            for p in range(n_process):
                file.write(str(proc_files_names[p]) + ', ' + str(model_files_names[match_idxs[p]]))
                file.write('\n')
        print("Done. Folder took %s seconds to process." % (time.time() - start_time))
        print('\n------------------')
        print('Processed ' + str(i+1) + ' out of ' + str(len(_folders_list)) + ' folders.')
        print('------------------')

    print('\nFinished!')




