import sys
import pyrallis
from src.deep.data_generator import DataConfig, DataGenerator


if __name__ == '__main__':

    if len(sys.argv) > 1:
        config = pyrallis.parse(DataConfig)
    else:
        config_path = './config/data_generation/bad_area.yml'
        config = pyrallis.parse(DataConfig, config_path=config_path)
    # main(config)
    DataGenerator(config).run()
