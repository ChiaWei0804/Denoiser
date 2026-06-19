# wujian@2018

import random
import torch as th
import numpy as np

from torch.utils.data.dataloader import default_collate
import torch.utils.data as dat

from audio import WaveReader


def make_dataloader(train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=4,
                    subset_ratio=1.0):
    dataset = Dataset(**data_kwargs)
    print('dataset full length: ', len(dataset))
    # 只取前 subset_ratio 比例的資料（例如 0.5 = 一半）
    if 0 < subset_ratio < 1.0:
        n = int(len(dataset) * subset_ratio)
        dataset = dat.Subset(dataset, list(range(n)))
        print(f'using subset: {n} / {len(dataset) + (len(dataset.dataset) - n)} utterances ({subset_ratio*100:.0f}%)')
    print(f'dataset[18] shape: ', dataset[18]['mix'].shape)
    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)


class Dataset(object):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp="", ref_scp=None, sample_rate=8000):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = [
            WaveReader(ref, sample_rate=sample_rate) for ref in ref_scp
        ]

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        ref = [reader[key] for reader in self.ref]
        return {
            "mix": mix.astype(np.float32),
            "ref": [r.astype(np.float32) for r in ref]
        }


class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "ref": [ndarray...]
        """
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = [ref[s:s + self.chunk_size] for ref in eg["ref"]]
        return chunk

    def split(self, eg):
        N = eg["mix"].size
        # too short, throw away
        if N < self.least:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = [
                np.pad(ref, (0, P), "constant") for ref in eg["ref"]
            ]
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


class DataLoader(object):
    """
    Online dataloader for chunk-level PIT
    """
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=16,
                 train=True):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)
        # just return batch of egs, support multiple workers
        self.eg_loader = dat.DataLoader(dataset,
                                        batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        shuffle=train,
                                        collate_fn=self._collate)

    def _collate(self, batch):
        """
        Online split utterances
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(chunk_list[s:s + self.batch_size])
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __len__(self):
        """
        Estimate total batches for progress bar display.
        """
        num_utts = len(self.eg_loader.dataset)
        # Sample one utterance to get actual length
        sample = self.eg_loader.dataset[0]
        N = sample["mix"].size
        chunk_size = self.splitter.chunk_size
        least = self.splitter.least
        if N < least:
            chunks_per_utt = 0
        elif N < chunk_size:
            chunks_per_utt = 1
        else:
            chunks_per_utt = max(1, (N - chunk_size) // least + 1)
        return (num_utts * chunks_per_utt) // self.batch_size

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj
