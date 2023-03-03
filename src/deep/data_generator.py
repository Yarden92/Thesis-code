

class DataGenerator:
    def __init__(self, data, batch_size, shuffle=True, augment=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()