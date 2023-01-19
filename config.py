import os
import torch
import sys

class Nih14:
    server = "agave"  # server = lab | psc | agave
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

        if self.server == "lab":
            self.data_root = "/mnt/dfs/jpang12/datasets/nih_xray14/images/images/"
            self.model_path = os.path.join("saved_models", self.method)
        elif self.server == "agave":
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




# class VinDRCXR:
#     server = "lab"  # server = lab | bridges2 | agave
#     debug_mode = False
#     patience= 20
#     epochs = 100
#     def __init__(self, args):
#         self.batch_size = args.batch_size
#         self.num_workers = args.num_workers
#         self.lr = args.lr
#         self.weight = args.weight
#         self.gpu = args.gpu
#         self.run = args.run
#         self.backbone = args.backbone
#         self.gt = args.gt
#         self.method = "VinDRCXR_{}_{}_run{}".format(self.backbone,self.gt, self.run)

#         if self.server == "lab":
#             self.data_root = "/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0"
#             self.model_path = os.path.join("saved_models", self.method)
#         elif self.server == "agave":
#             self.data_root = "/data/jliang12/jpang12/dataset/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0"
#             self.model_path = os.path.join("/data/jliang12/jpang12/VinDRCXR/saved_models",self.method)
#         elif self.server == "bridges2":
#             self.data_root = "/ocean/projects/bcs190005p/shared/jpang12/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0"
#             self.model_path = os.path.join("/ocean/projects/bcs190005p/jpang12/VinDRCXR/saved_models",self.method)


#         if 'global' in self.gt:
#             self.train_list = "data/vindrcxr/global/train_{}.txt".format(self.gt)
#             self.valid_list = "data/vindrcxr/global/test_global.txt"
#             self.test_list = "data/vindrcxr/global/test_global.txt"
#             self.num_classes = 6
#         elif 'local' in self.gt:
#             self.train_list = "data/vindrcxr/local/train_{}.txt".format(self.gt)
#             self.valid_list = "data/vindrcxr/local/test_local.txt"
#             self.test_list = "data/vindrcxr/local/test_local.txt"
#             self.num_classes = 22


#         if not os.path.exists(self.model_path):
#             os.makedirs(self.model_path)
#         logs_path = os.path.join(self.model_path, "Logs")
#         if not os.path.exists(logs_path):
#             os.makedirs(logs_path)

#         if os.path.exists(os.path.join(logs_path, "log.txt")):
#             self.log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
#         else:
#             self.log_writter = open(os.path.join(logs_path, "log.txt"), 'w')

#         if args.gpu is not None:
#             self.device = "cuda"
#         else:
#             self.device = "cpu"
#         if self.debug_mode:
#             self.log_writter = sys.stdout

#     def display(self):
#         """Display Configuration values."""
#         print("\nConfigurations:", file=self.log_writter)
#         for a in dir(self):
#             if not a.startswith("__") and not callable(getattr(self, a)):
#                 print("{:30} {}".format(a, getattr(self, a)), file=self.log_writter)
#         print("\n", file=self.log_writter)






# class ShenzhenRCXR:
#     server = "agave"  # server = lab | bridges2 | agave
#     debug_mode = False
#     patience= 50
#     epochs = 1000
#     def __init__(self, args):
#         self.batch_size = args.batch_size
#         self.num_workers = args.num_workers
#         self.lr = args.lr
#         self.weight = args.weight
#         self.gpu = args.gpu
#         self.run = args.run
#         self.backbone = args.backbone
#         self.method = "ShenzhenCXR_{}_run{}".format(self.backbone, self.run)

#         if self.server == "lab":
#             self.data_root = "/mnt/dfs/jpang12/datasets/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png"
#             self.model_path = os.path.join("saved_models", self.method)
#         elif self.server == "agave":
#             self.data_root = "/data/jliang12/jpang12/dataset/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png"
#             self.model_path = os.path.join("/data/jliang12/jpang12/ShenzhenCXR/saved_models",self.method)
#         elif self.server == "bridges2":
#             self.data_root = "/ocean/projects/bcs190005p/shared/jpang12/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png"
#             self.model_path = os.path.join("/ocean/projects/bcs190005p/jpang12/ShenzhenCXR/saved_models",self.method)


#         self.train_list = "data/shenzhen/ShenzenCXR_train_data.txt"
#         self.valid_list = "data/shenzhen/ShenzenCXR_valid_data.txt"
#         self.test_list = "data/shenzhen/ShenzenCXR_test_data.txt"
#         self.num_classes = 1


#         if not os.path.exists(self.model_path):
#             os.makedirs(self.model_path)
#         logs_path = os.path.join(self.model_path, "Logs")
#         if not os.path.exists(logs_path):
#             os.makedirs(logs_path)

#         if os.path.exists(os.path.join(logs_path, "log.txt")):
#             self.log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
#         else:
#             self.log_writter = open(os.path.join(logs_path, "log.txt"), 'w')

#         if args.gpu is not None:
#             self.device = "cuda"
#         else:
#             self.device = "cpu"
#         if self.debug_mode:
#             self.log_writter = sys.stdout

#     def display(self):
#         """Display Configuration values."""
#         print("\nConfigurations:", file=self.log_writter)
#         for a in dir(self):
#             if not a.startswith("__") and not callable(getattr(self, a)):
#                 print("{:30} {}".format(a, getattr(self, a)), file=self.log_writter)
#         print("\n", file=self.log_writter)


# class RSNAPneumonia:
#     server = "agave"  # server = lab | bridges2 | agave
#     debug_mode = False
#     patience= 50
#     epochs = 1000
#     def __init__(self, args):
#         self.batch_size = args.batch_size
#         self.num_workers = args.num_workers
#         self.lr = args.lr
#         self.weight = args.weight
#         self.gpu = args.gpu
#         self.run = args.run
#         self.backbone = args.backbone
#         self.method = "{}/run{}".format(self.backbone, self.run)

#         if self.server == "lab":
#             self.data_root = "/mnt/dfs/jpang12/datasets/rsna-pneumonia-detection-challenge/stage_2_train_images_png"
#             self.model_path = os.path.join("saved_models", self.method)
#         elif self.server == "agave":
#             self.data_root = "/data/jliang12/jpang12/dataset/rsna-pneumonia-detection-challenge/stage_2_train_images_png"
#             self.model_path = os.path.join("/data/jliang12/jpang12/RSNAPneumonia/saved_models",self.method)
#         elif self.server == "bridges2":
#             self.data_root = "/ocean/projects/bcs190005p/shared/jpang12/rsna-pneumonia-detection-challenge/stage_2_train_images_png"
#             self.model_path = os.path.join("/ocean/projects/bcs190005p/jpang12/RSNAPneumonia/saved_models",self.method)


#         self.train_list = "data/rsna_pneumonia/train.txt"
#         self.valid_list = "data/rsna_pneumonia/val.txt"
#         self.test_list = "data/rsna_pneumonia/test.txt"
#         self.num_classes = 1


#         if not os.path.exists(self.model_path):
#             os.makedirs(self.model_path)
#         logs_path = os.path.join(self.model_path, "Logs")
#         if not os.path.exists(logs_path):
#             os.makedirs(logs_path)

#         if os.path.exists(os.path.join(logs_path, "log.txt")):
#             self.log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
#         else:
#             self.log_writter = open(os.path.join(logs_path, "log.txt"), 'w')

#         if args.gpu is not None:
#             self.device = "cuda"
#         else:
#             self.device = "cpu"
#         if self.debug_mode:
#             self.log_writter = sys.stdout

#     def display(self):
#         """Display Configuration values."""
#         print("\nConfigurations:", file=self.log_writter)
#         for a in dir(self):
#             if not a.startswith("__") and not callable(getattr(self, a)):
#                 print("{:30} {}".format(a, getattr(self, a)), file=self.log_writter)
#         print("\n", file=self.log_writter)
