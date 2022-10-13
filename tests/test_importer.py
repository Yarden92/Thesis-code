print('test1: import src.deep.ml_ops.Trainer\t', end="")
try:
    from src.deep.trainers import Trainer

    print('[OK]')
except Exception as e:
    print(f'[FAIL] <- {e}')


print('test2: import torch\t\t\t', end="")
try:
    import torch
    print('[OK]')
except Exception as e:
    print(f'[FAIL] <- {e}')

print('test3: import ...src.deep.ml_ops.Trainer\t', end="")
try:
    print('[OK]')
except Exception as e:
    print(f'[FAIL] <- {e}')



