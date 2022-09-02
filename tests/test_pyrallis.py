from dataclasses import dataclass

import pyrallis


@dataclass
class ExampleConfig:
    arg1: str = 'asd'
    arg2: int = 2


if __name__ == '__main__':
    config = pyrallis.parse(config_class=ExampleConfig)
    print(f'arg1={config.arg1}, arg2={config.arg2}')
    print('done')
