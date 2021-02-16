import glob
import os
#from monai.transforms import Resized
import tables
#from monai.transforms import Affine, Rand3DElasticd, RandAffine, LoadNifti, Orientationd, Spacingd, LoadNiftid, AddChanneld, ScaleIntensityRanged, Resize
import pickle
from tqdm import tqdm
import numpy as np
#from copyData import electrodes

electrodes = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','O1','OZ','O2']

def fetch_training_data_files(path="data/original/"):
    training_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join(path, "*")):
        baseName=os.path.basename(subject_dir)
        tokens=baseName.split("_")
        if len(tokens)<3:
            continue
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in electrodes:
            subject_files.append(os.path.join(subject_dir, modality+ ".npy"))
        training_data_files.append(tuple(subject_files))
    print(training_data_files)
    return training_data_files, subject_ids

def getTruthData(subject_ids, labelPath):
    truthData = pickle_load(labelPath) 
    labels = []
    for subj in subject_ids:
        tokens = subj.split("_")
        patient_id = tokens[0]
        TMS= tokens[1]
        Type=tokens[2]
        trial = int(tokens[3][1:])
        key=(patient_id, TMS, trial)
        labels.append(truthData[key])
    return labels
        

def write_data_to_file(training_data_files, out_file, image_shape, subject_ids, truthData, normalize=True):
    n_samples = len(training_data_files)
    n_channels = 60

    try:
        hdf5_file, data_storage  = create_data_file(out_file,
                                                                                  n_channels=n_channels,
                                                                                  n_samples=n_samples,
                                                                                  image_shape=image_shape,
                                                                                  subject_ids =subject_ids,
                                                                                  truthData=truthData)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        #os.remove(out_file)
        print("hdf5 file creation failed")
        raise e

    write_image_data_to_file(training_data_files, data_storage, image_shape)
    print (hdf5_file)
    #if normalize:
    #    normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


def create_data_file(out_file, n_channels, n_samples, image_shape, subject_ids, truthData):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    label_shape = tuple([0, 1])
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    #truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.Float32Atom(), shape=label_shape,
    #                                        filters=filters, expectedrows=n_samples)
    #                                         filters=filters, expectedrows=n_samples)
    #                                                 filters=filters, expectedrows=n_samples)
    print("to be added")
    print(subject_ids)
    print(truthData)
    hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    hdf5_file.create_array(hdf5_file.root, 'truth', obj=truthData)
    print("created file:")
    print(hdf5_file)

    return hdf5_file, data_storage



def write_image_data_to_file(image_files, data_storage, image_shape):
    for set_of_files in tqdm(image_files):
        images = load_files(set_of_files)
        # subject_data = [image.get_data() for image in images]
        subject_data = images
        add_data_to_storage(data_storage, subject_data)
    return data_storage


def add_data_to_storage(data_storage, subject_data):
    data_storage.append(np.asarray(subject_data[:])[np.newaxis])


def load_files(in_files):
    data_list=[]
    for f in in_files:
        data=np.load(f)
        data_list.append(data)
    return data_list

    
"""
def reslice_image_set(in_files, image_shape):
    data_dict = {"image":in_files[0], "label":in_files[1]}
    loader = LoadNiftid(keys=("image", "label"))
    data_dict = loader(data_dict)
    original_image_size = data_dict["image"].shape
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)
    resize = Resized(["image", "label"], image_shape)
    data_dict = resize(data_dict)
    scaler = ScaleIntensityRanged(keys=["image"], a_min=30, a_max=130, b_min=0.0, b_max=1.0, clip=True) #30-130
    data_dict = scaler(data_dict)
    new_img_matrix, new_lbl_matrix = data_dict["image"][0], data_dict["label"][0]
    resized_img = nib.Nifti1Image(new_img_matrix, data_dict["image_meta_dict"]["affine"])
    resized_lbl = nib.Nifti1Image(new_lbl_matrix, data_dict["label_meta_dict"]["affine"])
    return ([resized_img, resized_lbl], original_image_size)
    #return None
"""


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file,  protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

"""
#original is located in prediction
def resize_to_original_size(image, original_size):
    original_size = [int(dim) for dim in original_size]
    resize = Resize(original_size, mode="nearest")
    resized_data = resize(image.get_fdata()[np.newaxis])
    resized_img = nib.Nifti1Image(resized_data[0], image.affine)
    return resized_img

def convert_data_file_indice_to_image(data_index, output_dir, data_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    original_size=  data_file.root.size[data_index]
    affine = data_file.root.affine[data_index]
    test_data = np.asarray([data_file.root.data[data_index]])

    image = nib.Nifti1Image(test_data[0, 0], affine)
    image = resize_to_original_size(image, original_size)
    image.to_filename(os.path.join(output_dir, "data_ct.nii.gz"))

    test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
    test_truth = resize_to_original_size(test_truth, original_size)
    test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))


def convert_data_file_to_image(image_indices, hdf5_file, output_dir="."):
    data_file = tables.open_file(hdf5_file, "r")
    if not image_indices:
        sample_num= len(data_file.root.data)
        image_indices = list(range(sample_num))
    
    for index in image_indices:
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "retrieved_case_{}".format(index))
        convert_data_file_indice_to_image(data_index=index, output_dir=case_directory, data_file=data_file)    
    data_file.close()

"""    