"""Pascal Transparent Semantic Segmentation Dataset."""
import os
import logging
import torch
import numpy as np

from PIL import Image
from .seg_data_base import SegmentationDataset


class SberSegmentationAllNoFU(SegmentationDataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = TransparentSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    # BASE_DIR = 'train/images/'
    BASE_DIR = ''
    NUM_CLASS = 12

    def __init__(self, root='datasets/Sber2400', split='test', mode=None, transform=None, **kwargs):
        super(SberSegmentationAllNoFU, self).__init__(root, split, mode, transform, **kwargs)
        root = os.path.join(self.root, self.BASE_DIR)
        assert os.path.exists(root), "Please put the data in {SEG_ROOT}/datasets/Sber2400"
        self.images, self.masks = _get_sber2400_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        logging.info('Found {} images in the folder {}'.format(len(self.images), root))

        ############ Important ##############
        # Change the palette for each dataset
        self.src_palette = Image.open(root+"all_no_fu_palette.png")
        self.src_palette = self.src_palette.convert("P", palette=Image.ADAPTIVE)
        ############ ######### ##############

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _val_sync_transform_resize(self, img, mask):
        short_size = self.crop_size
        img = img.resize(short_size, Image.BILINEAR)
        mask = mask.resize(short_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # mask = Image.open(self.masks[index]).convert("P", palette=Image.ADAPTIVE)
        mask = Image.open(self.masks[index]).quantize(palette=self.src_palette)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, resize=True)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._val_sync_transform_resize(img, mask)
        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ('Background', 'Glass', 'Mirror', 'Other optical surface', 'Floor',
                'Floor under obstacle', 'none_1' , 'none_2', 'none_3', 'none_4',
                'none_5', 'none_6')


def _get_sber2400_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'train/images')
        mask_folder = os.path.join(folder, 'train/Semantic Merged Floor')
    elif mode == "val":
        img_folder = os.path.join(folder, 'validation/images')
        mask_folder = os.path.join(folder, 'validation/Semantic Merged Floor')
    else:
        assert  mode == "test"
        img_folder = os.path.join(folder, 'test/images')
        mask_folder = os.path.join(folder, 'test/Semantic Merged Floor')

    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)

        if filename.endswith(".png"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                logging.info('cannot find the mask:', maskpath)

    return img_paths, mask_paths


if __name__ == '__main__':
    train_dataset = SberSegmentationAllNoFU()
