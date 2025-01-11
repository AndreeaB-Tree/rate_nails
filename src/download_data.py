from roboflow import Roboflow


def download_dataset():
    rf = Roboflow(api_key="wtTW8cLT8GIOUZ2VUSSh")
    project = rf.workspace("personal-projects-jfbag").project("nails_segmentation")
    version = project.version(44)
    dataset = version.download("yolov5")
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    print(dataset.location)
    print(f"Dataset downloaded to: {dataset.location}")

if __name__ == "__main__":
    download_dataset()
