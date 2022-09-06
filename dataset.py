import os
import pickle
from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset

# (original) prepare_data를 통해 lmdb key값을 어느 정도 맞춰놓음. -> 해당 부분 수정
class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            
            if not os.path.isfile(f"{path}/key_list.pickle"):
                self.key_list = list(txn.cursor().iternext(values=False))
                
                print("Successfully Generated a Key list!")

                with open(f"{path}/key_list.pickle", "wb") as f:
                    pickle.dump(self.key_list, f)

            else:
                with open(f"{path}/key_list.pickle", "rb") as f:
                    self.key_list = pickle.load(f)

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = self.key_list[index]
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        
        if self.transform:
            img = self.transform(img)

        return img
