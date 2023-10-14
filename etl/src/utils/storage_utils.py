from src.settings import BUCKET_NAME, FOLDER_NAME

from src.services.storage import s3


def list_folders(bucket_name=BUCKET_NAME, s3_folder_name=FOLDER_NAME):
    bucket = s3.Bucket(bucket_name)
    folders = set()

    for obj in bucket.objects.filter(Prefix=s3_folder_name):
        obj_key = obj.key
        folder = "/".join(obj_key.split("/")[:-1])
        if folder:
            folders.add(folder.split("/")[-1])
    return folders


def list_objects(s3_folder_name):
    bucket = s3.Bucket(BUCKET_NAME)
    objs = list(bucket.objects.filter(Prefix=f"{FOLDER_NAME}/{s3_folder_name}"))
    file_names = []

    for obj in objs:
        print("Found " + obj.key)
        out_name = obj.key.split("/")[-1]
        if out_name[-4:] == ".pdf":
            file_names.append(out_name)
            print(out_name)

    return file_names
