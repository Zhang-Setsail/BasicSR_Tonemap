from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, paired_paths_from_folder_with_different_ext
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import imageio
imageio.plugins.freeimage.download()
import numpy as np


@DATASET_REGISTRY.register()
class PairedHdrDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:

        dataroot_gt (str): Data root path for gt.                   Ground-Truth目录
        dataroot_lq (str): Data root path for lq.                   HDR目录
        meta_info_file (str): Path for meta information file.       元信息文件
        io_backend (dict): IO backend type and other kwarg.         IO后端类型和其他kwarg
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.                                          每个文件名的模板。请注意，模板不包括文件扩展名。
        gt_size (int): Cropped patched size for gt patches.         
        use_hflip (bool): Use horizontal flips.                     
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedHdrDataset, self).__init__()
        self.opt = opt

        '''
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        '''
        
        if 'non_gt_ext' in opt:
            self.non_gt_ext = opt['non_gt_ext']    
        else:
            self.non_gt_ext = '.hdr'

        self.gt_folder, self.hdr_folder = opt['dataroot_gt'], opt['dataroot_hdr']

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder_with_different_ext([self.hdr_folder, self.gt_folder], ['hdr', 'gt'], self.filename_tmpl, non_gt_ext = self.non_gt_ext)

    def __getitem__(self, index):
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        gt = imageio.imread(gt_path).astype('float32')

        hdr_path = self.paths[index]['hdr_path']
        if self.non_gt_ext == '.hdr':
            hdr = imageio.imread(hdr_path, format="HDR-FI").astype('float32')
        elif self.non_gt_ext == '.exr':
            hdr = imageio.imread(hdr_path, format="HDR-FI").astype('float32')
        else:
            raise TypeError(f'unable to read the file with extension: {self.non_gt_ext}')
        
        '''# # augmentation for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # random crop
        #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # # color space transform
        # if 'color' in self.opt and self.opt['color'] == 'y':
        #     img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
        #     img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # # TODO: It is better to update the datasets, rather than force to crop
        # if self.opt['phase'] != 'train':
        #     img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]'''

        # BGR to RGB, HWC to CHW, numpy to tensor
        gt, hdr = img2tensor([gt, hdr], bgr2rgb=True, float32=True)

        return {'lq': hdr, 'gt': gt, 'lq_path': hdr_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
