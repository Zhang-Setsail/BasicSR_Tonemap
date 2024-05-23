from torch.utils import data as data
# from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, paired_paths_from_folder_with_different_ext
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

lq_folder = "./datasets/VDS_Dataset/hdr/"
gt_folder = "./datasets/VDS_Dataset/gt/"
filename_tmpl = '{}'

paths = paired_paths_from_folder_with_different_ext([lq_folder, gt_folder], ['hdr', 'gt'], filename_tmpl, non_gt_ext='.hdr')
print(paths)