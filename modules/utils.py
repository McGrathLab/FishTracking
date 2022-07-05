import pandas as pd
import tarfile


tar = tarfile.open('../data/MC_singlenuc62_3_Tk65_060220/MC_singlenuc62_3_Tk65_060220_inference.tar')
for member in tar:
    f = tar.extractfile(member)
    try:
        frame = member.name.split('_')[-1].split('.')[0]
        dets = [l.decode('utf-8').strip() for l in f.readlines()]
    except AttributeError:
        pass
tar.close()




