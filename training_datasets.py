import cv2
import numpy as np
from os.path import join, basename, splitext, exists
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset
from annotator.hed import HEDdetector
from annotator.util import HWC3
from annotator.midas import MidasDetector

class TrainingDataset(Dataset):
    def __init__(self, rootdir: str, prompt: str):
        """
        We assume there is only one prompt in this case.
        """
        self.rootdir = rootdir
        self.imgfns = glob(join(rootdir, '*.jpg'))
        self.prompt = prompt
        self.detect = None
        # check whether we already have the HED data
        self.generate_detectmaps()

    def get_detectmap_filename(self, fn: str) -> str:
        rootfn, ext = splitext(fn)
        return rootfn + '_detectmap.png'

    def generate_detectmap(self, imgfn: str, outfn: str):
        if self.detect is None:
            self.detect = self.get_detector()
        input_image = cv2.imread(imgfn)
        H, W, C = input_image.shape
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = HWC3(input_image)
        detected_map = self.detect(input_image)
        if type(detected_map) == tuple:
            detected_map = detected_map[0]
        detected_map = HWC3(detected_map)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = cv2.cvtColor(detected_map, cv2.COLOR_RGB2BGR)
        cv2.imwrite(outfn, detected_map)

    def get_detector(self):
        raise NotImplementedError()

    def generate_detectmaps(self):
        for fn in tqdm(self.imgfns, desc="Generating detectmaps"):
            hedfn = self.get_detectmap_filename(fn)
            if not exists(hedfn):
                self.generate_detectmap(fn, hedfn)

    def __len__(self):
        return len(self.imgfns)

    def __getitem__(self, idx):
        target_filename = self.imgfns[idx]
        source_filename = self.get_detectmap_filename(target_filename)
        prompt = self.prompt

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

class HEDDataset(TrainingDataset):
    def get_detectmap_filename(self, fn: str) -> str:
        rootfn, ext = splitext(fn)
        return rootfn + '_hed.png'

    def get_detector(self):
        return HEDdetector()

class DepthDataset(TrainingDataset):
    def get_detectmap_filename(self, fn: str) -> str:
        rootfn, ext = splitext(fn)
        return rootfn + '_depth.png'
    
    def get_detector(self):
        return MidasDetector()

if __name__ == '__main__':
    dataset = HEDDataset('/home/grapeot/co/Dreambooth-Anything/data/Lycoris', 'an anime in LycorisAnime style')
    print(len(dataset))

    item = dataset[1234]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    print(jpg.shape)
    print(hint.shape)

    dataset = DepthDataset('/home/grapeot/co/Dreambooth-Anything/data/Lycoris', 'an anime in LycorisAnime style')
    print(len(dataset))

    item = dataset[1234]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    print(jpg.shape)
    print(hint.shape)