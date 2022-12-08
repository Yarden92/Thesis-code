class FolderTypes:
    Data = 'data'
    Analysis = 'analysis'
    LinuxShit = 'linux_sh'
    Unknown = 'unknown'

class DataMethods:
    @staticmethod
    def check_folder_type(sub_name):
        if sub_name.startswith('_'): return FolderTypes.Analysis
        if sub_name.startswith('.'): return FolderTypes.LinuxShit
        if '=' not in sub_name: return FolderTypes.Unknown
        return FolderTypes.Data