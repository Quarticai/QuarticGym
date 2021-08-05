import os


def parent_dir_and_name(file_path):
    """
    >>> file_path="a/b.c"
    >>> parent_dir_and_name(file_path)
    ('/root/.../a', 'b.c')
    :param file_path:
    :return:
    """
    return os.path.split(os.path.abspath(file_path))


def get_things_in_loc(in_path, just_files=True):
    """
    in_path can be file path or dir path.
    This function return a list of file paths
    in in_path if in_path is a dir, or within the 
    parent path of in_path if it is not a dir.
    just_files=False will let the function go recursively
    into the subdirs.
    """
    # TODO: check for file
    if not os.path.exists(in_path):
        print(str(in_path) + " does not exists!")
        return
    re_list = []
    if not os.path.isdir(in_path):
        in_path = parent_dir_and_name(in_path)[0]

    for name in os.listdir(in_path):
        name_path = os.path.abspath(os.path.join(in_path, name))
        if os.path.isfile(name_path):
            re_list.append(name_path)
        elif not just_files:
            if os.path.isdir(name_path):
                re_list += get_things_in_loc(name_path, just_files)
    return re_list


def normalize_spaces(space, max_space=None, min_space=None):
    """
    normalize each column of observation/action(e.g. Sugar feed rate) to be in [-1,1] such that it looks like a Box
    and space can be the whole original space (X by D) or just one row in the original space (D,)
    :param space: numpy array
    """
    assert not isinstance(space, list)
    if max_space is None:
        max_space = space.max(axis=0)
    if min_space is None:
        min_space = space.min(axis=0)
    gap = max_space - min_space
    gap += 1e-8 # to avoid div by 0
    full_sum = max_space + min_space
    return (2 * space - full_sum) / gap, max_space, min_space


def denormalize_spaces(space_normalized, max_space=None, min_space=None):
    """
    same as above, and space_normalized can be the whole normalized original space or just one row in the normalized space
    """
    assert not isinstance(space_normalized, list)
    if max_space is None:
        max_space = space_normalized.max(axis=0)
    if min_space is None:
        min_space = space_normalized.min(axis=0)
    gap = max_space - min_space
    gap += 1e-8 # to avoid div by 0
    full_sum = max_space + min_space
    return (space_normalized * gap + full_sum) / 2, max_space, min_space

