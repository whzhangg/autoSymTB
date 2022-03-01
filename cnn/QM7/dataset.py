from torch.utils.data import Dataset


class InputData(Dataset):
    """a very basic dataset class"""
    def __init__(self, datalist) -> None:
        self.datas = datalist

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]


if __name__ == "__main__":
    pass
    