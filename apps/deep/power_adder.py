import os
import json
import numpy as np
from tqdm import tqdm

from src.deep.data_methods import DataMethods


class PowerAdder:

    def __init__(self, is_overright=True, verbose_level=0) -> None:
        self.is_overright = is_overright
        self.verbose_level = verbose_level  # 0 - no prints, 1 - print only dataset name, 2 - print everything

    def add_to_all(self, datasets_dir='/data/yarcoh/thesis_data/data/datasets'):
        """Add average powers to all the datasets in the datasets directory."""
        dirs = os.listdir(datasets_dir)
        if self.verbose_level == 0:
            dirs = tqdm(dirs, desc='adding average powers to all datasets')

        for dataset in dirs:
            dataset_path = os.path.join(datasets_dir, dataset)
            if self.verbose_level == 0:
                dirs.set_description(f'adding average powers to dataset: {dataset}')

            self.calculate_and_add_powers(dataset_path)
            
            if self.verbose_level > 0:
                print(f'added average powers to dataset: {dataset}')

    def calculate_and_add_powers(self, dataset_path='/data/yarcoh/thesis_data/data/datasets/qam16_50x300'):
        """Calculate and add average powers to the config files in each subdirectory."""

        subdirs = os.listdir(dataset_path)
        if self.verbose_level == 0:
            subdirs = tqdm(subdirs, desc='main loop - on subdirs')

        # Iterate over the subdirectories in the dataset
        for subdir in subdirs:
            # verify that the subdirectory is a directory
            if DataMethods.check_folder_type(subdir) != DataMethods.FolderTypes.Data:
                continue
            
            subdir_path = os.path.join(dataset_path, subdir)

            if not self.is_overright and self._is_power_exists(subdir_path):
                if self.verbose_level > 1:
                    print(f'average powers already exist for subdir: {subdir}')
                continue
            
            # Get the list of data files in the subdirectory
            dir_list = os.listdir(subdir_path)
            if self.verbose_level == 0:
                dir_list = tqdm(dir_list, desc=f'subloop0 - collecting files @{subdir}')
            x_data_files, y_data_files = [], [] 
            for filename in dir_list:
                if filename.endswith('x.npy'):
                    x_data_files.append(filename)
                elif filename.endswith('y.npy'):
                    y_data_files.append(filename)


            # x_data_files = [file for file in os.listdir(subdir_path) if file.endswith('x.npy')]
            # y_data_files = [file for file in os.listdir(subdir_path) if file.endswith('y.npy')]

            # Calculate the average power for x and y samples
            avg_x_power = 0.0
            avg_y_power = 0.0
            if self.verbose_level == 0:
                x_data_files = tqdm(x_data_files, desc=f'subloop1 - calculating power for each file @{subdir}')
            for x_data_file, y_data_file in zip(x_data_files, y_data_files):
                x_data = np.load(os.path.join(subdir_path, x_data_file))
                y_data = np.load(os.path.join(subdir_path, y_data_file))

                avg_x_power += self._calculate_average_power(x_data)
                avg_y_power += self._calculate_average_power(y_data)

            avg_x_power /= len(x_data_files)
            avg_y_power /= len(y_data_files)

            avg_x_power_dBm = self.linear_to_dBm(avg_x_power)
            avg_y_power_dBm = self.linear_to_dBm(avg_y_power)

            # Add the average powers to the config file
            self._add_power_to_subdir(subdir_path, avg_x_power, avg_y_power)
            if self.verbose_level > 1:
                print((f'average power for subdir: {subdir} is:'
                       f'\n\tavg_x_power: {avg_x_power_dBm} dBm'
                       f'\n\tavg_y_power: {avg_y_power_dBm} dBm'))

            if self.verbose_level == 0:
                subdirs.set_description(f'subdir {subdir}: x_avg of {avg_x_power:.2f}')

    def remove_powers_from_subdir(self, subdir_path):
        """Remove the average powers from the config file in a subdirectory."""
        config_path = os.path.join(subdir_path, 'conf.json')

        with open(config_path, 'r+') as config_file:
            config = json.load(config_file)
            config.pop('avg_x_power', None)
            config.pop('avg_y_power', None)

            # Move the file pointer to the beginning of the file
            config_file.seek(0)
            json.dump(config, config_file, indent=4)
            config_file.truncate()

    def replace_key_in_config_files(self, root_folder):
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename == 'conf.json':
                    file_path = os.path.join(foldername, filename)
                    with open(file_path, 'r') as file:
                        config = json.load(file)
                    
                    if 'ssf' in config and 'h' in config['ssf']:
                        config['ssf']['dz'] = config['ssf'].pop('h')

                        with open(file_path, 'w') as file:
                            json.dump(config, file, indent=4)
                        if self.verbose_level > 1:
                            print(f'replaced key in config file: {file_path}')
            if self.verbose_level > 0:
                print(f'finished with folder: {foldername}')
            

    def _calculate_average_power(self, data):
        """Calculate the average power of a complex numpy array."""
        return np.mean(np.abs(data) ** 2)

    def _add_power_to_subdir(self, subdir_path, avg_x_power, avg_y_power):
        """Write the average powers to the config file in a subdirectory."""
        config_path = os.path.join(subdir_path, 'conf.json')

        with open(config_path, 'r+') as config_file:
            config = json.load(config_file)
            config['avg_x_power'] = avg_x_power
            config['avg_y_power'] = avg_y_power

            # Move the file pointer to the beginning of the file
            config_file.seek(0)
            json.dump(config, config_file, indent=4)
            config_file.truncate()
    def linear_to_dBm(self, linear):
        """Convert a linear value to dBm."""
        return 10 * np.log10(linear) + 30

    def _is_power_exists(self, subdir):
        """Check if the average powers are already in the config file."""
        config_path = os.path.join(subdir, 'conf.json')

        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return 'avg_x_power' in config and 'avg_y_power' in config


if __name__ == '__main__':
    # dataset_path = '/data/yarcoh/thesis_data/data/datasets/qam16_50x300'
    # calculate_and_add_powers(dataset_path, verbose = True)

    datasets_dir = '/data/yarcoh/thesis_data/data/datasets'
    power_adder = PowerAdder(is_overright=False, verbose_level=2)
    # power_adder.add_to_all(datasets_dir)
    power_adder.replace_key_in_config_files(datasets_dir)
