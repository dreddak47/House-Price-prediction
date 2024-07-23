import os
import shutil

path_for_images = "Canvas_renamed_resized/"
P = sorted(os.listdir(path_for_images))
P = [i for i in P if i[-1]=='g']

f = open("Indices_of_Test_data")
test_indices = f.readlines()
test_indices = [int(i) for i in test_indices]
f.close()

out_path_train = "Data/train/images/"
out_path_test = "Data/test/images/"

os.makedirs(out_path_train, exist_ok=True)
os.makedirs(out_path_test, exist_ok=True)

for i in range(len(P)):
    image_name = P[i]
    image_index = image_name.split('_')[0]
    if int(image_index) in test_indices:
        shutil.copy(path_for_images+P[i],out_path_test)
    else:
        shutil.copy(path_for_images+P[i],out_path_train)
        

