from src.deep.data_loaders import create_channels
from src.optics.config_manager import ChannelConfig, ConfigManager


class PowerAdder2:

    def __init__(self,
                 num_repetitions: int = 10,  # Number of repetitions to estimate average power
                 overwrite: bool = False  # If True, overwrite existing average power in config file
                 ) -> None:
        self.num_repetitions = num_repetitions
        self.overwrite = overwrite

    # ------------------------------ actions ------------------------------ #
    # - calc_single_mu()

    def calc_single_mu(self, subdir_path) -> float:
        # reads the config,
        # loop pa_config.num_repetitions times:
        # regenerate io samples, and calculate signal power
        # return the average power
        cs_config: ChannelConfig = ConfigManager.read_config(subdir_path)
        in_cs, _ = create_channels(cs_config)

        avg_power = 0
        for i in range(self.num_repetitions):
            dbm_power = in_cs.simulate_and_calc_power()
            avg_power += dbm_power
        avg_power /= self.num_repetitions
        
        return avg_power

        

    # ------------------------------ private ------------------------------ #



if __name__ == '__main__':
    pa = PowerAdder2()
    pa.calc_single_mu('/data/yarcoh/thesis_data/data/datasets/b1/30000samples_15mu_W01/mu=0.10')
