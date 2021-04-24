import os
 
def rename_jpg(main_path, i):
    jpg_list = os.listdir(main_path)
    for jpg in jpg_list:
        if jpg.endswith(".jpg"):
            src = os.path.join(main_path, jpg)
            new_name = str(i) + ".jpg"
            i = i + 1
            dst = os.path.join(main_path, new_name)
            print(src)
            print(dst)
            os.rename(src, dst)
 
 
if __name__ == "__main__":
    rename_jpg(r"./dog", 0)
    rename_jpg(r"./baby", 0)
    rename_jpg(r"./people", 0)

