import os
import random
from datetime import datetime
from collections import defaultdict
from torch.utils.data import Dataset
import torch
import rasterio
import numpy as np
from torchvision.transforms.functional import resize
import re

class WildfireDataset(Dataset):
    def __init__(self, satellite, countries, image_size=256, years=range(2017,2024), transform_resize=None, random_crop=True, terratorch_baselines=False):
        """
        Initialize the dataset for multiple countries and years.

        Parameters:
            base_dir (str): The base directory containing the files.
            satellite (str): The satellite identifier (e.g., 's2').
            countries (list): A list of country identifiers (e.g., ['CA', 'US']).
            years (iterable): A range or list of years to consider.
            image_size (int, optional): Desired size to which images will be cropped or resized.
            transform_resize (callable, optional): A function/transform to apply to the images.
            random_crop (bool, optional): Whether to apply a random crop (ensuring consistency for pre/post images & labels).
        """
        # self.base_dir = base_dir
        self.satellite = satellite
        self.countries = countries if isinstance(countries, list) else [countries]  # Support list or single string
        # self.years = years
        self.image_size = image_size
        self.transform_resize = transform_resize
        self.random_crop = random_crop
        self.crop_coords = {}  # Store crop coordinates for consistency
        # self.skip_count = 0
        self.terratorch_baselines = terratorch_baselines

        # Aggregate data from all countries and years
        self.data = []
        for country in self.countries:
            # if country == "US":
            #     years = years # range(2017, 2023)
            # elif country == "CA":
            #     years = range(2017, 2023)
            for year in years:
                if country == "US":
                    # year_dir = os.path.join(base_dir, f"wildfires_{satellite}_{country}_{year}") # I changed this when I downloaded the new data to common place
                    base_dir = f'/geoinfo_proj/Shared/wildfire_data/{satellite}/US_final/'
                    year_dir = os.path.join(base_dir, f"US_{year}")
                elif country == "CA": # CA
                    base_dir = '/home/a/a/aadelow/data/'
                    year_dir = os.path.join(base_dir, f"wildfires_{satellite}_{country}_{year}")

                if os.path.exists(year_dir):
                    grouped_files = self._group_files_by_event_id(self._list_files(year_dir))
                    self.data.extend(self._prepare_data(year_dir, grouped_files, country))

        self.id_to_index = {}
        for idx, (pre_img, post_img, pre_lbl, post_lbl) in enumerate(self.data):
            # Extract the fire ID from any one of the paths
            stem = os.path.basename(pre_img).split("_")
            if stem[0] == "CA":
                fire_id = "_".join(stem[:4])  # e.g. CA_2017_AB_1213
            elif stem[0] == "US":
                # e.g. US_2017_AR3565309368420170320_S2_pre_...
                part = pre_img.split("_S2_")[0]
                fire_id = os.path.basename(part)
            else:
                continue
            self.id_to_index[fire_id] = idx


    def _list_files(self, directory):
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def _group_files_by_event_id(self, files):
        grouped_files = defaultdict(list)
        for file in files:
            parts = file.split('_')
            if len(parts) >= 4:
                event_id = f"{parts[2]}_{parts[3]}"
                grouped_files[event_id].append(file)
        return grouped_files

    def _prepare_data(self, directory, grouped_files, country):
        data = []
        for event_id, files in grouped_files.items():
            pre_label, post_label, pre_image, post_image = self._extract_pre_post_data(files, country)
            if pre_label and post_label and pre_image and post_image:
                data.append((
                    os.path.join(directory, pre_image),
                    os.path.join(directory, post_image),
                    os.path.join(directory, pre_label),
                    os.path.join(directory, post_label)
                ))
            # if not post_label or not pre_image or not post_image:
            #     self.skip_count += 1
        return data
    
    def filter_by_states(self, states, inplace=True):
        """
        Filters the dataset to include only entries from the specified states.

        Parameters:
            states (list or set): List of state identifiers to include (e.g., ['AB', 'BC']).
            inplace (bool): Whether to modify self.data in place. If False, returns filtered list.

        Returns:
            Optional[list]: Filtered list of data if inplace=False.
        """
        if not isinstance(states, (list, set)):
            raise ValueError("`states` must be a list or set of state abbreviations.")

        if self.countries == ["CA"]:
            pattern = re.compile(r'CA_\d{4}_([A-Z]+)_')
        elif self.countries == ["US"]:
            pattern = re.compile(r'US_\d{4}_([A-Z]{2})\d+_')
        elif self.countries == ["CA", "US"] or self.countries == ["US", "CA"]:
            pattern = re.compile(r'(?:CA|US)_\d{4}_([A-Z]{2,})')

        def extract_state_from_path(path):
            match = pattern.search(path)
            if match:
                return match.group(1)
            return None

        filtered_data = []
        for item in self.data:
            path = item[0]  # only need one file from the tuple
            state = extract_state_from_path(path)
            if state in states:
                filtered_data.append(item)

        if inplace:
            self.data = filtered_data
        else:
            return filtered_data
        
    def _extract_state_year_from_path(self, path):
        """
        Extract (year, state) from a file path for CA/US datasets.
        Returns (year:int or None, state:str or None).
        """
        # Choose a regex depending on countries configured
        if self.countries == ["CA"]:
            # e.g., CA_2019_AB_...
            pattern = re.compile(r'CA_(\d{4})_([A-Z]+)_')
        elif self.countries == ["US"]:
            # e.g., US_2020_CA123... or US_2020_CA_...
            # More permissive: capture two-letter state after year, up to next underscore
            pattern = re.compile(r'US_(\d{4})_([A-Z]{2})[^_]*_')
        else:
            # Mixed CA/US
            pattern = re.compile(r'(?:CA|US)_(\d{4})_([A-Z]{2,})')

        m = pattern.search(path)
        if not m:
            return None, None
        year = int(m.group(1))
        state = m.group(2)
        return year, state


    def filter_by_state_years(self, states, years, mode="include", inplace=True):
        """
        Keep or drop entries matching (state, year) pairs.

        Parameters:
            states (list|set): state/province codes to match (e.g., ['CA','OR'] or ['BC','AB']).
            years (int|list|set|tuple): a single year or multiple years (e.g., 2020 or [2019,2020,2021]).
            mode (str): "include" to KEEP matches; "exclude" to DROP matches.
            inplace (bool): if True, modifies self.data; else returns a filtered copy.

        Returns:
            Optional[list]: filtered data if inplace=False.
        """
        if not isinstance(states, (list, set, tuple)):
            raise ValueError("`states` must be a list/set/tuple of state abbreviations.")
        # normalize
        states_set = set(states)

        if isinstance(years, int):
            years_set = {years}
        elif isinstance(years, (list, set, tuple)):
            years_set = {int(y) for y in years}
        else:
            raise ValueError("`years` must be an int or an iterable of ints.")

        mode = mode.lower()
        if mode not in {"include", "exclude"}:
            raise ValueError("`mode` must be either 'include' or 'exclude'.")

        filtered = []
        for item in self.data:
            path = item[0]  # just inspect one of the paths in the tuple
            yr, st = self._extract_state_year_from_path(path)

            is_match = (yr in years_set) and (st in states_set)

            if (mode == "include" and is_match) or (mode == "exclude" and not is_match):
                filtered.append(item)

        if inplace:
            self.data = filtered
        else:
            return filtered


    def _extract_pre_post_data_CA(self, file_list):
        pre_label, post_label, pre_image, post_image = None, None, None, None
        label_files = [f for f in file_list if 'label' in f]
        date_files = [f for f in file_list if 'label' not in f]

        # Extract label files
        for file in label_files:
            if 'pre_label' in file:
                pre_label = file
            elif 'post_label' in file:
                post_label = file

        label_year = None
        if pre_label and post_label:
            label_year = int(pre_label.split('_')[1])

        date_dict = {}
        for file in date_files:
            try:
                date_str = file.split('_')[-1].split('.')[0]
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                date_dict[date_obj] = file
            except ValueError:
                continue

        sorted_dates = sorted(date_dict.keys())

        if sorted_dates and label_year:
            pre_year_dates = [d for d in sorted_dates if d.year == (label_year - 1)]
            if pre_year_dates:
                pre_image = date_dict[max(pre_year_dates)]

            post_year_dates = [d for d in sorted_dates if d.year == label_year]
            if post_year_dates:
                post_image = date_dict[min(post_year_dates)]

        return pre_label, post_label, pre_image, post_image

    def _extract_pre_post_data_US(self, file_list):
        pre_label, post_label, pre_image, post_image = None, None, None, None
        label_files = [f for f in file_list if 'label' in f]
        date_files = [f for f in file_list if 'label' not in f]

        # Extract label files
        for file in label_files:
            if 'pre_label' in file:
                pre_label = file
            elif 'post_label' in file:
                post_label = file

        
        for file in date_files:
            if '_pre_' in file and 'rank1' in file:
                pre_image = file
            elif '_post_' in file and 'rank1' in file:
                post_image = file

        return pre_label, post_label, pre_image, post_image

        # pre_label, post_label, pre_image, post_image = None, None, None, None
        # label_files = [f for f in file_list if 'label' in f]
        # date_files = [f for f in file_list if 'label' not in f]

        # # Extract label files
        # for file in label_files:
        #     if 'pre_label' in file:
        #         pre_label = file
        #     elif 'post_label' in file:
        #         post_label = file

        # pre_dates = {}
        # post_dates = {}

        # for file in date_files:
        #     try:
        #         date_str = file.split('_')[-1].split('.')[0]
        #         date_obj = datetime.strptime(date_str, '%Y%m%d')

        #         if '_pre_' in file:
        #             pre_dates[date_obj] = file
        #         elif '_post_' in file:
        #             post_dates[date_obj] = file

        #     except ValueError:
        #         continue

        # if pre_dates:
        #     pre_image = pre_dates[max(pre_dates.keys())]  # Latest pre image
        # if post_dates:
        #     post_image = post_dates[min(post_dates.keys())]  # Earliest post image

        # return pre_label, post_label, pre_image, post_image

    def _extract_pre_post_data(self, file_list, country):
        if country == "CA":
            return self._extract_pre_post_data_CA(file_list)
        elif country == "US":
            return self._extract_pre_post_data_US(file_list)
        else:
            raise ValueError(f"Unsupported country: {country}")

    def _generate_crop_coords(self, h, w):
        """
        Generate random top-left coordinates for cropping.
        """
        top = random.randint(0, max(0, h - self.image_size))
        left = random.randint(0, max(0, w - self.image_size))
        return top, left

    def _apply_crop(self, tensor_data, top, left):
        """
        Apply the same crop to all images and labels.
        """
        return tensor_data[:, top:top + self.image_size, left:left + self.image_size]

    def load_tif_as_tensor(self, filepath, is_label=False, crop_coords=None):
        """
        Loads a .tif file and applies transformations (resize or crop).
        Ensures that pre/post images and labels are cropped consistently.
        """
        with rasterio.open(filepath) as src:
            data = src.read()  # (C, H, W)
            nodata_value = src.nodata

        if is_label:
            tensor_data = torch.tensor(data, dtype=torch.float32)
            tensor_data = (tensor_data > 0).float()
        else:
            selected_bands = data[:3]  # Use the first 3 bands
            selected_bands = np.nan_to_num(selected_bands, nan=0.0)
            if nodata_value is not None:
                selected_bands[selected_bands == nodata_value] = 0
            tensor_data = torch.tensor(selected_bands, dtype=torch.float32)
            tensor_data = torch.clamp(tensor_data, min=0, max=5000) / 5000.0  # Normalize

        # Get image dimensions
        _, h, w = tensor_data.shape

        if self.transform_resize:
            tensor_data = resize(tensor_data, [self.image_size, self.image_size])
        elif self.random_crop and crop_coords:
            top, left = crop_coords
            tensor_data = self._apply_crop(tensor_data, top, left)

        return tensor_data

    def __getitem__(self, idx):
        pre_image_path, post_image_path, pre_label_path, post_label_path = self.data[idx]

        # Ensure the same random crop for all images in this sample
        if idx not in self.crop_coords:
            with rasterio.open(pre_image_path) as src:
                _, h, w = src.read().shape  # Get original dimensions
            self.crop_coords[idx] = self._generate_crop_coords(h, w)

        crop_coords = self.crop_coords[idx]

        pre_image = self.load_tif_as_tensor(pre_image_path, is_label=False, crop_coords=crop_coords)
        post_image = self.load_tif_as_tensor(post_image_path, is_label=False, crop_coords=crop_coords)
        pre_label = self.load_tif_as_tensor(pre_label_path, is_label=True, crop_coords=crop_coords)
        post_label = self.load_tif_as_tensor(post_label_path, is_label=True, crop_coords=crop_coords)

        if self.terratorch_baselines:
            return {
                # "pre_image": pre_image,
                # "post_image": post_image,
                # "pre_label": pre_label,
                # "post_label": post_label,
                # "paths": [pre_image_path, post_image_path, pre_label_path, post_label_path],
                "image": post_image,
                "mask": post_label.squeeze(0).long(),
            }
        else:
            return {
                "pre_image": pre_image,
                "post_image": post_image,
                "pre_label": pre_label,
                "post_label": post_label,
                "paths": [pre_image_path, post_image_path, pre_label_path, post_label_path],
                # "image": post_image,
                # "mask": post_label.squeeze(0).long(),
            }

    def __len__(self):
        return len(self.data)
    
    def get_by_id(self, fire_id, load_images=True):
        """
        Retrieve a dataset sample (or file paths) by its fire ID.
        If load_images=False, returns paths only without opening rasters.
        """
        if not hasattr(self, "id_to_index"):
            # build lookup table on first use
            self.id_to_index = {}
            for idx, (pre_img, post_img, pre_lbl, post_lbl) in enumerate(self.data):
                name = os.path.basename(pre_img)
                if name.startswith("CA_"):
                    fid = "_".join(name.split("_")[:4])
                elif name.startswith("US_"):
                    fid = name.split("_S2_")[0]
                else:
                    continue
                self.id_to_index[fid] = idx

        idx = self.id_to_index.get(fire_id)
        if idx is None:
            raise KeyError(f"❌ Fire ID {fire_id} not found in dataset.")

        return self.__getitem__(idx) if load_images else self.data[idx]
    
    def filter_by_fire_ids(self, fire_ids, inplace=True):
        fire_ids = set(fire_ids)
        filtered = []
        for item in self.data:
            path = item[0]
            name = os.path.basename(path)
            
            if name.startswith("CA_"):
                fid = "_".join(name.split("_")[:4])
            elif name.startswith("US_"):
                fid = name.split("_S2_")[0]
            else:
                continue

            if fid in fire_ids:
                filtered.append(item)

        if inplace:
            self.data = filtered
        else:
            return filtered
        
    def filter_by_ids_and_years(self, fire_ids, years, inplace=True):
        """
        Filter dataset to keep only samples whose fire ID AND year match.
        
        Parameters:
            fire_ids (list|set): fire IDs to keep
            years (int|list|set): years to keep
            inplace (bool): modify dataset in place (default)
        """
        fire_ids = set(fire_ids)

        if isinstance(years, int):
            years = {years}
        else:
            years = {int(y) for y in years}

        filtered = []

        for item in self.data:
            path = item[0]   # use pre-image path
            base = os.path.basename(path)

            # --- Extract fire ID ---
            if base.startswith("CA_"):
                fid = "_".join(base.split("_")[:4])
            elif base.startswith("US_"):
                fid = base.split("_S2_")[0]
            else:
                continue

            if fid not in fire_ids:
                continue

            # --- Extract year ---
            yr, _ = self._extract_state_year_from_path(path)
            if yr in years:
                filtered.append(item)

        if inplace:
            self.data = filtered
        else:
            return filtered

    def get_all_fire_ids(self):
        """
        Returns a sorted list of all unique fire IDs in the dataset.
        """
        fire_ids = []

        for item in self.data:
            path = item[0]  # pre-image path
            name = os.path.basename(path)

            if name.startswith("CA_"):
                fid = "_".join(name.split("_")[:4])
            elif name.startswith("US_"):
                fid = name.split("_S2_")[0]
            else:
                continue

            fire_ids.append(fid)

        return sorted(set(fire_ids))
