import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image

# ==================================================================
# CÁC HÀM TIỆN ÍCH CHUNG (GIỮ NGUYÊN TỪ REPO GỐC)
# ==================================================================

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``."""
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory."""
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items

# ==================================================================
# CÁC LỚP CẤU TRÚC DỮ LIỆU CƠ BẢN (GIỮ NGUYÊN TỪ REPO GỐC)
# ==================================================================

class Datum:
    """
    Đại diện cho một mẫu dữ liệu (data instance) với các thuộc tính cơ bản.
    """
    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """
    Lớp cơ sở thống nhất cho tất cả các bộ dữ liệu.
    Chứa các phương thức chung để xử lý và tạo các tập con few-shot.
    """
    dataset_dir = ''
    domains = []

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x
        self._train_u = train_u
        self._val = val
        self._test = test

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test
        
    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=True):
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')
        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)
            output.append(dataset)

        if len(output) == 1:
            return output[0]
        return output

    def split_dataset_by_label(self, data_source):
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)
        return output

# ==================================================================
# LỚP WRAPPER ĐÃ ĐƯỢC SỬA ĐỔI ĐỂ TƯƠNG THÍCH VỚI TRANSFORMERS
# ==================================================================

class DatasetWrapper(TorchDataset):
    """
    Lớp Wrapper để chuyển đổi một data_source (danh sách các Datum)
    thành một đối tượng Dataset mà PyTorch DataLoader có thể sử dụng.
    """
    def __init__(self, data_source, input_size, transform=None, is_train=False, **kwargs):
        self.data_source = data_source
        self.transform = transform # Đây sẽ là processor.image_processor
        self.is_train = is_train
        self.input_size = input_size

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        """
        Lấy một mẫu (sample) từ dataset và xử lý nó.
        """
        item = self.data_source[idx]

        # 1. Đọc ảnh từ đường dẫn dưới dạng ảnh PIL
        img0 = read_image(item.impath)

        # 2. Kiểm tra xem transform (processor) có tồn tại không
        if self.transform is None:
            raise ValueError("Transform (processor) không được cung cấp cho DatasetWrapper.")
            
        # 3. Áp dụng transform (processor) cho ảnh PIL
        # Đầu ra là một đối tượng kiểu dictionary (BatchFeature).
        processed_output = self.transform(images=img0, return_tensors="pt")

        # 4. Trích xuất tensor 'pixel_values' và loại bỏ chiều batch thừa
        img_tensor = processed_output['pixel_values'].squeeze(0)

        # 5. Trả về một tuple (tensor_ảnh, nhãn)
        return img_tensor, item.label

# ==================================================================
# HÀM BUILDER CHO DATALOADER (GIỮ NGUYÊN TỪ REPO GỐC)
# ==================================================================

def build_data_loader(
    data_source=None,
    batch_size=64,
    input_size=224,
    tfm=None,
    is_train=True,
    shuffle=False,
    dataset_wrapper=None,
    num_workers=8
):
    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available())
    )
    assert len(data_loader) > 0

    return data_loader