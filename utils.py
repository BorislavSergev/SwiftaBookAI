import os

def check_directories(folder):
    expected_dirs = ['heart', 'oblong', 'oval', 'round', 'square']
    actual_dirs = os.listdir(folder)
    for d in expected_dirs:
        if d not in actual_dirs:
            return False
    return True

def log_directory_structure(folder):
    for root, dirs, files in os.walk(folder):
        print(f"Root: {root}, Dirs: {dirs}, Files: {files}")
