import os


def extract_filename(file):
    return os.path.basename(file)


def extract_base_filename(file):
    basename = os.path.basename(file)
    return '.'.join(basename.split('.')[:-1])


def extract_file_extension(file):
    basename = os.path.basename(file)
    return '.' + str(basename.split('.')[-1])


def change_file_extension(file, new_ext):
    parts = file.split('.')
    parts[-1] = new_ext[1:]
    return '.'.join(parts)


def parent_dir(file):
    path = os.path.abspath(os.path.join(os.path.dirname(file), os.pardir))
    return path
