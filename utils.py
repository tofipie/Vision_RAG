import os
def get_data_files():
    data_files = []
    for dirname, _, filenames in os.walk("images"):
        for filename in filenames:
            data_files.append(os.path.join(filename))
    return data_files
