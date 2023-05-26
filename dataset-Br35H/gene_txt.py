import glob

jpgs = glob.glob('./*/*.jpg')[120:]

for x in jpgs:
    with open('train.txt', 'a') as f:
        f.write('dataset-Br35H/' + x[2:] + '\n')
