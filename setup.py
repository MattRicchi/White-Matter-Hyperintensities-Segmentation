#Always run this to ensure you have all functions on your path

def Setup_Script():
    import sys
    import subprocess
    import os

    sys.path.append(os.getcwd())
    print(os.getcwd())

    print('Checking you hve all packages needed for the codebase...')

    try:
        import numpy
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
    try:
        import nibabel
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nibabel'])
    try:
        import tqdm
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])
    try:
        import tensorflow
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    try:
        import focal_loss
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'focal-loss'])
    
    print('Using cpu: ' + platform.processro())
    print('All good, off we go!')

    return