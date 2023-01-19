import os
import torch
import sys

class Nih14:
    server = "agave"  # server = agave
    debug_mode = False
    patience= 50 # was 50
    epochs = 1000
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.weight = args.weight
        self.gpu = args.gpu
        self.run = args.run
        self.backbone = args.backbone
        self.method = "nih14_{}_run{}".format(self.backbone, self.run)

        if self.server == "agave":
            self.data_root = "/scratch/nuislam/ChestXRay14_images/" #"/data/jliang12/jpang12/data/nih_xray14/images/images/"
            self.model_path = os.path.join("saved_models",self.method)

        self.train_list = "data/nih_xray14/official/train_official.txt"
        self.valid_list = "data/nih_xray14/official/val_official.txt"
        self.test_list = "data/nih_xray14/official/test_official.txt"

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        logs_path = os.path.join(self.model_path, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        if os.path.exists(os.path.join(logs_path, "log.txt")):
            self.log_writter = sys.stdout # open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            self.log_writter = sys.stdout # open(os.path.join(logs_path, "log.txt"), 'w')

        # self.tsne_path = os.path.join(logs_path, "tsne")
        # if not os.path.exists(self.tsne_path):
        #     os.makedirs(self.tsne_path)

        if args.gpu is not None:
            self.device = "cuda"
        else:
            self.device = "cpu"
        if self.debug_mode:
            self.log_writter = sys.stdout

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:", file=self.log_writter)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)), file=self.log_writter)
        print("\n", file=self.log_writter)
