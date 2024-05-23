import yaml

from basicsr.data.hdr_pair_dataset import PairedHdrDataset


def test_hdrimagedataset():
    """Test dataset: PairedHdrDataset"""

    opt_str = r"""
name: Test
type: PairedImageDataset
dataroot_gt: ./datasets/VDS_Dataset/gt/
dataroot_hdr: ./datasets/VDS_Dataset/hdr/
filename_tmpl: '{}'


gt_size: 128

phase: train
"""
    opt = yaml.safe_load(opt_str)
    print(opt)

    dataset = PairedHdrDataset(opt)

    # test __getitem__
    result = dataset.__getitem__(0)
    # # check returned keys
    # expected_keys = ['lq', 'gt', 'lq_path', 'gt_path']
    # assert set(expected_keys).issubset(set(result.keys()))
    # # check shape and contents
    # assert result['gt'].shape == (3, 128, 128)
    # assert result['lq'].shape == (3, 32, 32)
    # assert result['lq_path'] == 'tests/data/lq/baboon.png'
    # assert result['gt_path'] == 'tests/data/gt/baboon.png'

    # # ------------------ test filename_tmpl -------------------- #
    # opt.pop('filename_tmpl')
    # opt['io_backend'] = dict(type='disk')
    # dataset = PairedImageDataset(opt)
    # assert dataset.filename_tmpl == '{}'

    # # ------------------ test scan folder mode -------------------- #
    # opt.pop('meta_info_file')
    # opt['io_backend'] = dict(type='disk')
    # dataset = PairedImageDataset(opt)
    # assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    # assert len(dataset) == 2  # whether to correctly scan folders

    # # ------------------ test lmdb backend and with y channel-------------------- #
    # opt['dataroot_gt'] = 'tests/data/gt.lmdb'
    # opt['dataroot_lq'] = 'tests/data/lq.lmdb'
    # opt['io_backend'] = dict(type='lmdb')
    # opt['color'] = 'y'
    # opt['mean'] = [0.5]
    # opt['std'] = [0.5]

    # dataset = PairedImageDataset(opt)
    # assert dataset.io_backend_opt['type'] == 'lmdb'  # io backend
    # assert len(dataset) == 2  # whether to read correct meta info
    # assert dataset.std == [0.5]

    # # test __getitem__
    # result = dataset.__getitem__(1)
    # # check returned keys
    # expected_keys = ['lq', 'gt', 'lq_path', 'gt_path']
    # assert set(expected_keys).issubset(set(result.keys()))
    # # check shape and contents
    # assert result['gt'].shape == (1, 128, 128)
    # assert result['lq'].shape == (1, 32, 32)
    # assert result['lq_path'] == 'comic'
    # assert result['gt_path'] == 'comic'

    # # ------------------ test case: val/test mode -------------------- #
    # opt['phase'] = 'test'
    # opt['io_backend'] = dict(type='lmdb')
    # dataset = PairedImageDataset(opt)

    # # test __getitem__
    # result = dataset.__getitem__(0)
    # # check returned keys
    # expected_keys = ['lq', 'gt', 'lq_path', 'gt_path']
    # assert set(expected_keys).issubset(set(result.keys()))
    # # check shape and contents
    # assert result['gt'].shape == (1, 480, 492)
    # assert result['lq'].shape == (1, 120, 123)
    # assert result['lq_path'] == 'baboon'
    # assert result['gt_path'] == 'baboon'

test_hdrimagedataset()