import os

output_path = '/home/g9liveyourdream/UDA/MDD/output'
accc = 0.8567
acc=round(accc,2)


path = os.path.join("output_path", str(acc*100)+"_model.pth.tar")

print(path)

