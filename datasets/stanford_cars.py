import os
from scipy.io import loadmat

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase


template = [
    'a photo of a {}.',
    'a photo of the car {}.',
    'a photo of my new car, a {}.',
    'a picture of a {} car.',
    'a side view of the {}.',
    'a photo of the front of a {}.',
    'the {} model of car.'
]


class StanfordCars(DatasetBase):

    dataset_dir = 'StanfordCars'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)['annotations'][0]
        meta_file = loadmat(meta_file)['class_names'][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]['fname'][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]['class'][0, 0]
            label = int(label) - 1 # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(' ')
            year = names.pop(-1)
            names.insert(0, year)
            classname = ' '.join(names)
            item = Datum(
                impath=impath,
                label=label,
                classname=classname
            )
            items.append(item)
        
        return items