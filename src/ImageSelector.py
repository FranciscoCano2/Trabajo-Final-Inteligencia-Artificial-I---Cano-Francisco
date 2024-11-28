import os

class ImageSelector:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = [
            {"index": i, "path": os.path.join(folder_path, filename)}
            for i, filename in enumerate(os.listdir(folder_path)) 
            if filename.endswith(".jpg") or filename.endswith(".png")
        ]

    def get_image_path_by_index(self, index):
        for img in self.images:
            if img["index"] == index:
                return img["path"]
        return None

    def list_images(self):
        return [{"index": img["index"], "path": img["path"]} for img in self.images]
