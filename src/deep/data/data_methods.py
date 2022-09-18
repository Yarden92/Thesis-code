
class DataMethods:
    @staticmethod
    def is_valid_subfolder(sub_name):
        if sub_name.startswith('_'): return False
        if sub_name.startswith('.'): return False
        if '=' not in sub_name: return False
        return True