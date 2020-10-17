from fastai.torch_core import *
from fastai.data_block import *
from fastai.basic_data import *
from fastai.tabular.data import emb_sz_rule
from preprocess import preprocess_data, get_data_fields

class DataProvider():
    def __init__(self, param_file:Path, cat_mode:int=1, norm='z', cache_root=None, segmentor=None):
        self.__files, self.__data  = {}, {}
        with open(param_file) as f:
            self.params = json.load(f)
        self.cache_root, self.segmentor, self.cat_mode, self.norm = cache_root, segmentor, cat_mode, norm

    def load_data(self, f):
        return _load_sepsis_data(f, self.segmentor, params=self.params, cache_root=self.cache_root, cat_mode=self.cat_mode, norm=self.norm)

    def preprocess(self, root:Path=Path('./'), lazy=True, max_workers=None):
        fnames = list(root.rglob('*.psv')) if root.is_dir() else [root]
        self.__files = {f.name:f for f in fnames}
        if lazy: return self
        
        max_workers = ifnone(max_workers, defaults.cpus//2)    
        if max_workers > 1 and len(list(fnames)) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_load_sepsis_data, fname, self.segmentor, params=self.params, cache_root=self.cache_root, cat_mode=self.cat_mode, norm=self.norm) for fname in fnames]
                for f in progress_bar(concurrent.futures.as_completed(futures), total=len(fnames)): pass
                self.__data = {fnames[i].name: f.result() for i, f in enumerate(futures)}
        else: self.__data = {f.name: _load_sepsis_data(f, self.segmentor, params=self.params, cache_root=self.cache_root, cat_mode=self.cat_mode) for f in fnames}
    
        return self

    def update_input(self, key:str, raw_data:pd.DataFrame):
        self.__files[key] = raw_data
        self.__data[key] = _load_sepsis_data(raw_data, self.segmentor, params=self.params, cache_root=self.cache_root, cat_mode=self.cat_mode, norm=self.norm)


    def __getitem__(self,key:str)->Tuple[pd.DataFrame, Tuple[int, int]]:
        f = self.__files[key]
        if key not in self.__data: 
            self.__data[key] = _load_sepsis_data(f, self.segmentor, self.params, self.cache_root, cat_mode=self.cat_mode, norm=self.norm)
        return self.__data[key]

    def __len__(self)->int: 
        return len(self.__files)

    @property
    def cache(self): 
        return self.__files 
    
    @property
    def data_fields(self):
        return get_data_fields(self.params, self.cat_mode)
    
    @property
    def cat_sizes(self):
        params = self.params
        cat_levels = params['cat_levels']
        sizes = []
        for cat in get_data_fields(params)[2]:
            level = cat_levels[cat] if cat in cat_levels else params['cat_convert_levels']
            sizes += [(level, emb_sz_rule(level))]
        return sizes

    
def _load_sepsis_data(f, segmentor, params=None, cache_root:Path=None, cat_mode:int=1, norm='z'):
    if isinstance(f, pd.DataFrame):
        data = f.copy()
        if params is not None: preprocess_data(data, params=params, cat_mode=cat_mode, norm=norm)

    else:
        cache_file = None if cache_root is None else cache_root.joinpath(Path(f.parent.name).joinpath(f.name))
        if cache_root is not None and cache_file.exists():
            data = pd.read_csv(cache_file, sep='|')
        else:
            data = pd.read_csv(f, sep='|')
            if params is not None: preprocess_data(data, params=params, cat_mode=cat_mode, norm=norm)
            if cache_root is not None:
                os.makedirs(cache_file.parent, exist_ok=True)
                data.to_csv(cache_file, sep='|', index=False)

    times, conts, cats = get_data_fields(params)
    data[times] = data[times].astype(np.float32, copy=False)
    data[conts] = data[conts].astype(np.float32, copy=False)
    data[cats]  = data[cats].astype(np.long, copy=False)

    if 'SepsisLabel' in data: data['SepsisLabel'].astype(np.long, copy=False)
    segs = None if segmentor is None else segmentor(len(data))
    return data, segs


def segment(n: int, window:int=8, slide:int=4, keep_last=True):
    segs = []
    if n == 0: return []
    elif n < window: raise Exception(f'Data length {n} is less than required {window}')
    elif n == window: segs = [[0, n]]
    else:
        start, stop = 0, window
        while stop < n:
            segs   += [[start, stop]]
            start  += slide
            stop    = start + window
        if keep_last and stop >= n: 
            segs   += [[n-window, n]]

    return segs


class SepsisPreprocessor(PreProcessor):
    def __init__(self, provider, target:str=None, progress=True):
        """ 
        target: [None, whole, steps, both] 
        """
        self.provider, self.target, self.progress = provider, target, progress
        self.y_processor_ = None

    def y_processor(self, target:str):
        if self.y_processor_ is None:
            self.y_processor_ = SepsisPreprocessor(self.provider, target=target, progress=self.process)
        return self.y_processor_

    def process(self, ds:Collection):
        if self.progress: pbar = progress_bar(range(len(ds.items)), auto_update=False, leave=False, parent=None)
        items = []        
        for i, file in enumerate(ds.items):
            data, segs = self.provider[file.name]
            if segs is not None:
                for start, stop in segs:
                    items += [self.process_one((file, (start, stop)))]
            else:   items += [self.process_one((file, (0, len(data))))]
            # if np.any(data.isna()): print(f'WARN: NA found in {file}')
            if self.progress: pbar.update(i+1)
        ds.items = items
    
    def process_one(self, item:Any):
        file, seg = item if isinstance(item, tuple) else (item, None)
        loader = self.load_data if self.target is None else self.load_target
        return SepsisItem(file, loader, seg=seg)

    def load_data(self, fname:PathOrStr, seg:Tuple[int, int]):
        data, _ = self.provider[fname.name]
        partials = [data[fields] for fields in self.provider.data_fields]
        if seg is not None:
            for i, part in enumerate(partials):
                partials[i] = partials[i].loc[seg[0]: seg[1]-1, :]
            
        return [tensor(part.values) for part in partials]

    def load_target(self, item:PathOrStr, seg:Tuple[int, int]):
        data, _ = self.provider[item.name]
        data = data['SepsisLabel']
        dev_sepsis = 1 if np.any(data > 0) else 0
        data = data if seg is None else data.loc[seg[0]: seg[1]-1]
        if self.target == 'steps': data = data.values
        elif self.target == 'both': data = (data.values, dev_sepsis)
        else: raise Exception(f'Unsupported target {self.target}')        
        return tensor(data) if self.target != 'both' else (tensor(data[0]), tensor(data[1]))

    @classmethod
    def _pad_steps(cls, seq, target, after=True, pad=torch.zeros):
        """seq has be in the shape (steps, *)"""
        mask = seq.size(0)
        mask = slice(None, mask) if after else slice(-mask, None)
        data = pad(*(target, *seq.size()[1:]), dtype=seq.dtype)
        data[mask] = seq
        return data

    def collate_sequence(self, batch, pad=torch.zeros, after=True, with_mask=True, max_steps:int=None):
        res, sizes = [], []
        for x, y in batch:
            x, y = to_data(x), to_data(y)
            sizes += [x[0].size(0)]
            res += [(x, y)]
        if pad is None: return torch.utils.data.dataloader.default_collate(to_data(res))

        batch = res
        res, max_seq = [], np.max(sizes) if max_steps is None else max_steps
        for ind, (x, y) in enumerate(batch):
            steps = sizes[ind]
            if steps < max_seq:
                x = [self.__class__._pad_steps(part, max_seq, after=after, pad=pad) for part in x]
                if self.target == 'steps':
                    y = self.__class__._pad_steps(y, max_seq, after=after, pad=pad)
                elif self.target == 'both':
                    y = (self.__class__._pad_steps(y[0], max_seq, after=after, pad=pad), y[1])

            if with_mask:
                mask = steps if after else -steps
                x = (*x, torch.as_tensor(mask))
                if self.target == 'steps': y = (y, torch.as_tensor(mask))
                elif self.target == 'both': y = (*y, torch.as_tensor(mask))
            res += [(x, y)]
        
        return torch.utils.data.dataloader.default_collate(to_data(res))


class SepsisItem(ItemBase):
    def __init__(self, fname:Path, loader:callable, seg:Tuple[int, int]=None):
        self.fname, self.loader, self.seg  = fname, loader, seg
        self.data_ = None

    def __str__(self): 
        name = self.fname.name
        name = name if self.seg is None else f'{name}[{self.seg[0]}-{self.seg[1]}]'
        name = f'{name}: Lazy' if self.data_ is None else f'{name}: {self.data_}'
        return name

    @property
    def data(self)->Tensor:
        if self.data_ is None: self.data_ = self.loader(self.fname, self.seg)
        return self.data_

    def __len__(self):
        return self.seg[1] - self.seg[0]


class EqualLenSampler(Sampler):
    def __init__(self, data_source:NPArrayList, max_batch=6, len_of=len):
        self.data_source, self.max_batch, self.len_of = data_source, max_batch, len_of
        self.lens = np.asarray([self.len_of(self.data_source[i]) for i in range(len(self.data_source))])
        self.len__ = None
        
    def __iter__(self):
        amount = len(self.lens)
        indices = np.arange(amount)
        ranges = list(set(self.lens.tolist()))

        #ranges.sort(reverse=True)
        ranges = np.random.permutation(ranges)
        
        for r in ranges:
            b_indices = np.random.permutation(indices[self.lens == r])
            i = 0
            while (i + self.max_batch) < len(b_indices):
                yield b_indices[i: i+self.max_batch]
                i += self.max_batch
            if i < self.max_batch: 
                yield b_indices[i:]
        
    def __len__(self) -> int: 
        if self.len__ is None:
            amount = len(self.lens)
            indices = np.arange(amount)
            ranges = list(set(self.lens.tolist()))
            counter = 0
            for r in ranges:
                b_indices = np.random.permutation(indices[self.lens == r])
                i = 0
                while (i + self.max_batch) < len(b_indices):
                    counter +=1
                    i += self.max_batch
                if i < self.max_batch: 
                    counter += 1
            self.len__ = counter
        return self.len__



class SepsisSampler(Sampler):
    "Go through the text data by order of length."

    def __init__(self, data_source:NPArrayList, key:KeyFunc): self.data_source,self.key = data_source,key
    def __len__(self) -> int: return len(self.data_source)
    def __iter__(self):
        lens = np.asarray([self.key(i) for i in range(len(self.data_source))])
        ranges = list(set(lens.tolist()))
        len_positions = []
        for r in ranges[::-1]:  # introduce a little bit randomness only in the group of same lengths.
            len_positions += [np.random.permutation(np.where(lens == r)[0])]
        len_positions = np.concatenate(len_positions)
        return iter(len_positions)

class SesisDatabunch(DataBunch):
    
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds:Optional[Dataset]=None, collate_fn=None,
               bs:int=32, val_bs:int=None, sort_seq=False, path:PathOrStr='.',
               device: torch.device = None, no_check:bool=False, provider=None, **dl_kwargs):
        if sort_seq:
            datasets = cls._init_ds(train_ds, valid_ds, test_ds)
            val_bs = ifnone(val_bs, bs)
            len_of = lambda x: len(x[0])
            train_dl = DataLoader(datasets[0], shuffle=False, batch_size=1, batch_sampler=EqualLenSampler(train_ds, max_batch=bs, len_of=len_of), drop_last=None, **dl_kwargs)
            dataloaders = [train_dl]
            for ds in datasets[1:]:
                if len(ds) <= 1: 
                    print(f'ignore {ds}')
                    continue
                dataloaders.append(DataLoader(ds, shuffle=False, batch_size=1, drop_last=None, batch_sampler=EqualLenSampler(ds, max_batch=bs, len_of=len_of), **dl_kwargs))
            return cls(*dataloaders, path=path, device=device, collate_fn=collate_fn, no_check=no_check)
        else: return super().create(train_ds, valid_ds, test_ds=test_ds, collate_fn=collate_fn, bs=bs, val_bs=val_bs, device=device,path=path)

    @property
    def cat_sizes(self):
        params = self.provider.params
        cat_levels = params['cat_levels']
        sizes = []
        for cat in get_data_fields(params)[2]:
            level = cat_levels[cat] if cat in cat_levels else params['cat_convert_levels']
            sizes += [(level, emb_sz_rule(level))]
        return sizes

    @property
    def data_fields(self):
        return get_data_fields(self.provider.params)


class SepsisItemList(ItemList):
    _bunch = SesisDatabunch

    @classmethod
    def from_dataset_home(cls, ds_home:Path, splitter, cat_mode:int, param_file:PathOrStr='./params.json', norm='z', segmentor=None, cache_root=None, full_seq=False,
                          preferred_steps=None, lazy=False, target='both', pc=None, **kwargs):
        provider = DataProvider(param_file, cat_mode,
            norm=norm,
            cache_root=cache_root, 
            segmentor=None if full_seq else segmentor
        ).preprocess(root=ds_home, lazy=lazy, max_workers=None if 'num_workers' not in kwargs else kwargs['num_workers'])

        processor = SepsisPreprocessor(provider, progress=True)
        y_processor = processor.y_processor(target=target)
        x_lists = cls.from_folder(ds_home, extensions=".psv", processor=processor)
        if pc is not None:
            print(f'WARN: only use {pc*100: .02f}% of data for debugging')
            x_lists = x_lists.filter_by_rand(p=pc)

        x_lists = x_lists.split_by_valid_func(splitter)
        xy_lists = x_lists.label_from_lists(x_lists.train, x_lists.valid, label_cls=ItemList, processor=y_processor)

        # if full_seq: kwargs['collate_fn'] = partial(y_processor.collate_sequence, with_mask=True)

        bunch = xy_lists.databunch(**kwargs)
        bunch.__setattr__('provider', provider)
        return bunch


def predict_segmented_items(model:nn.Module, items: Collection[SepsisItem], device:torch.device=defaults.device):
    batch = [[i for i in item.data] for item in items]
    batch = torch.utils.data.dataloader.default_collate(to_data(batch))
    batch = to_device(batch, device)
    values = model(*batch)

    segs = [item.seg for item in items]
    t_probs, t_preds = torch.zeros(segs[-1][-1], device=device), torch.zeros(segs[-1][-1], dtype=torch.long, device=device)
    for i, (seg, value) in enumerate(zip(segs, values)):
        probs, preds = value.softmax(dim=1).max(dim=1)
        t_index = torch.arange(seg[0], seg[1], device=device)
        mask = probs.ge(t_probs[t_index])
        t_probs[t_index[mask]] = probs[mask]
        t_preds[t_index[mask]] = preds[mask]

    return t_preds, t_probs


def _parallel(func, arr:Collection, max_workers:int=None, pbar=None):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, defaults.cpus)
    if max_workers<2: results = [func(o,i) for i,o in progress_bar(enumerate(arr), total=len(arr), parent=pbar, leave=(pbar is not None))]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
            results = []
            for f in progress_bar(concurrent.futures.as_completed(futures), total=len(arr), parent=pbar, leave=(pbar is not None)): results.append(f.result())
    if any([o is not None for o in results]): return results