from __future__ import division

from utils.models import *
from utils.utils import *
from utils.datasets import *

import operator

from PIL import Image

from torch.autograd import Variable
#from CostumeMatching import Costumematching
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
#from inception_inference import clothes_class_19
#from seg_inference import run_seg
#from CostumeMatching import Costumematching
#from detect_delete import clothes_class_detection
import lib.models.cls_hrnet as hrnet
from lib.config import config
from lib.config import update_config
#import tornado.ioloop
#import tornado.web
import sys
import time
import datetime
import random

class HisClothesClassify():
    def __init__(self, save_path):
        self.yolo = None
        self.hrnet = None
        self.small_class_lable = None
        self.config = config
        cfg_path = "./config/con_path.yaml"
        update_config(self.config, cfg_path)
        self.save_path = save_path
        self.yolo_size = 416
        self.yolo_conf_thres = 0.85
        self.yolo_nms_thres= 0.4
        self.big_label = ['upclothes', 'downclothes', 'dress', 'shoes', 'cap']
        self.small_label = ['boots', 'baseballcap', 'dcoat', 'fshirt', 'fshoes', 'gallus', 'hat', 'highheels', 'jumpsuit',
                   'lcoat', 'ldress', 'lshoes', 'lskirt', 'mshirt', 'pants', 'polo', 'scoat', 'sdress', 'shirt',
                   'shorts', 'sshoes', 'sskirt', 'suit-coat', 'suit-pants', 'sweater', 'tshirt', 'wsweater']

    def initModel(self):
        # load yolo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set up model
        self.yolo = Darknet(self.config.YOLO_MODEL_DEF, self.yolo_size).to(device)
        if self.config.YOLO_WEIGHT_PATH.endswith(".weights"):
            # Load darknet weights
            self.yolo.load_darknet_weights(self.config.YOLO_WEIGHT_PATH)
        else:
            # Load checkpoint weights
            self.yolo.load_state_dict(torch.load(self.config.YOLO_WEIGHT_PATH))
        self.yolo.eval()

        # load HRNet
        # cudnn related setting
        cudnn.benchmark = self.config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = self.config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = self.config.CUDNN.ENABLED
        self.hrnet = hrnet.get_cls_net(self.config)
        if self.config.TEST.MODEL_FILE:
            self.hrnet.load_state_dict(torch.load(self.config.TEST.MODEL_FILE))
        else:
            self.hrnet.load_state_dict(torch.load(self.config.HR_WEIGHT_PATH))
        gpus = list(self.config.GPUS)
        self.hrnet = torch.nn.DataParallel(self.hrnet, device_ids=gpus).cuda()
        self.hrnet.eval()

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # translate the image into tensor
        pil_img = Image.open(self.config.INIT_IMG_PATH)
        img_tran = transforms.ToTensor()(pil_img)
        img_tran = resize(img_tran, 416)
        img_tran = Variable(torch.unsqueeze(img_tran, dim=0).float(), requires_grad=False)
        img_tran = Variable(img_tran.type(Tensor))

        # Get detections
        detections = []
        with torch.no_grad():
            detections = self.yolo(img_tran)
            # free image
            torch.cuda.empty_cache()
            detections = self.yolo(img_tran)
            # free image
            torch.cuda.empty_cache()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        input = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])(pil_img)
        input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)
        # switch to evaluate mode
        cls_pred = 0
        with torch.no_grad():
            output = self.hrnet(input)
            # free image
            torch.cuda.empty_cache()


    def clothesDetection(self, img_path):

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        img_detections = []  # Stores detections for each image index

        # translate the image into tensor
        pil_img = Image.open(img_path)
        pil_img = pil_img.convert('RGB')
        img_tran = transforms.ToTensor()(pil_img)
        img_tran = resize(img_tran, 416)
        img_tran = Variable(torch.unsqueeze(img_tran, dim=0).float(), requires_grad=False)
        img_tran = Variable(img_tran.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = self.yolo(img_tran)
            # free image
            torch.cuda.empty_cache()
            # duplicate removal
            detections = non_max_suppression(detections, self.yolo_conf_thres, self.yolo_nms_thres)

        img_detections.extend(detections)
        detection = img_detections[0]

        img = np.array(pil_img)
        w, h = img.shape[:2]

        # duplicate bbox removal
        if detection is not None:
            # Rescale boxes to original image
            detection = rescale_boxes(detection, self.yolo_size, img.shape[:2])
            i = 0
            j = 0
            m = []
            detection_object = detection
            # print(detection[0])
            for x0, y0, x0_1, y0_1, conf_1, cls_conf_1, cls_pred_1 in detection_object:
                # detection=detection[torch.arange(detection.size(0))!=i]
                j = 0
                for x00, y00, x00_1, y00_1, conf_2, cls_conf_2, cls_pred_2 in detection_object:
                    if i != j:
                        if x0 >= x00:
                            if x00_1 <= x0:
                                j = j + 1
                                continue;
                            elif y0 >= y00:
                                if y0 >= y00_1:
                                    j = j + 1
                                    continue;
                                # zuo xia jiao
                                else:
                                    inter_area = ((x00_1 - x0) * (y00_1 - y0)) / (
                                            ((y0_1 - y0) * (x0_1 - x0)) + ((y00_1 - y00) * (x00_1 - x00)) - (
                                            x00_1 - x0) * (y00_1 - y0))
                                    # print('inter_area:%.4f\n' %(inter_area))
                                    if inter_area >= 0.80:
                                        if cls_conf_1 >= cls_conf_2:
                                            m.append(j)

                                            # detection=detection[torch.arange(detection.size(0))!=j]
                                            # print('num:%d  inter_area:%.4f  %s  %.4f\n' %(j, inter_area,classes[int(cls_pred_1)],cls_conf_2))

                                        else:
                                            m.append(i)
                                            # detection=detection[torch.arange(detection.size(0))!=i]
                                            # print('num:%d  inter_area:%.4f  %s  %.4f\n' %(i,inter_area,classes[int(cls_pred_1)],cls_conf_1))

                                    else:
                                        j = j + 1
                                        continue;
                            elif y00 >= y0_1:
                                j = j + 1
                                continue;
                            # zuo shang jiao
                            else:
                                inter_area = ((x00_1 - x0) * (y0_1 - y00)) / (
                                        ((y0_1 - y0) * (x0_1 - x0)) + ((y00_1 - y00) * (x00_1 - x00)) - (x00_1 - x0) * (
                                        y0_1 - y00))
                                # print('inter_area:%.4f\n' %(inter_area))
                                if inter_area >= 0.80:
                                    if cls_conf_1 >= cls_conf_2:
                                        m.append(j)
                                        # detection=detection[torch.arange(detection.size(0))!=j]
                                        # print('num:%d  inter_area:%.4f  %s  %.4f\n' %(j,inter_area,classes[int(cls_pred_2)],cls_conf_2))

                                    else:
                                        m.append(i)
                                        # detection=detection[torch.arange(detection.size(0))!=i]
                                        # print('num:%d   inter_area:%.4f  %s   %.4f\n' %(i,inter_area,classes[int(cls_pred_1)],cls_conf_1))

                                else:
                                    j = j + 1
                                    continue;

                        elif x00 >= x0_1:
                            j = j + 1
                            continue;
                        elif y0 >= y00:
                            if y0 >= y00_1:
                                j = j + 1
                                continue;
                            # you  xia jiao
                            else:  #
                                inter_area = ((x0_1 - x00) * (y00_1 - y0)) / (
                                        ((y0_1 - y0) * (x0_1 - x0)) + ((y00_1 - y00) * (x00_1 - x00)) - (x0_1 - x00) * (
                                        y00_1 - y0))
                                # print('inter_area:%.4f\n' %(inter_area))
                                if inter_area >= 0.80:
                                    if cls_conf_1 >= cls_conf_2:
                                        m.append(j)
                                        # detection=detection[torch.arange(detection.size(0))!=j]
                                        # print('num:%d  inter_area:%.4f  %s  %.4f\n' %(j,inter_area,classes[int(cls_pred_2)],cls_conf_2))

                                    else:
                                        m.append(i)
                                        # detection=detection[torch.arange(detection.size(0))!=i]
                                        # print('num:%d  inter_area:%.4f  %s  %.4f\n' %(i,inter_area,classes[int(cls_pred_1)],cls_conf_1))

                                else:
                                    j = j + 1
                                    continue;
                        elif y00 >= y0_1:
                            j = j + 1
                            continue;
                        # you shang jiao
                        else:
                            inter_area = ((x0_1 - x00) * (y0_1 - y00)) / (
                                    ((y0_1 - y0) * (x0_1 - x0)) + ((y00_1 - y00) * (x00_1 - x00)) - (x0_1 - x00) * (
                                    y0_1 - y00))
                            # print('inter_area:%.4f\n' %(inter_area))
                            if inter_area >= 0.80:
                                if cls_conf_1 >= cls_conf_2:
                                    m.append(j)
                                    # detection=detection[torch.arange(detection.size(0))!=j]
                                    # print('num:%d  inter_area:%.4f  %s   %.4f\n' %(j,inter_area,classes[int(cls_pred_2)],cls_conf_2))

                                else:
                                    m.append(i)
                                    # detection=detection[torch.arange(detection.size(0))!=i]
                                    # print('num:%d  inter_area:%.4f  %s   %4f\n' %(i,inter_area,classes[int(cls_pred_1)],cls_conf_1))

                            else:
                                j = j + 1
                                continue;
                    else:
                        j = j + 1
                        continue;
                    j = j + 1
                i = i + 1

            numbers = [int(x) for x in m]
            numbers = list(set(numbers))
            numbers.sort(reverse=True)
            for i in numbers:
                detection = detection[torch.arange(detection.size(0)) != i]

            return detection, w, h

    # only detect the cloth in the middle of image
    def clothesClassify(self, img_path):
        # prev_time = time.time()
        # img_path = "image/"+img_path
        detection, w, h = self.clothesDetection(img_path)

        # current_time = time.time()
        # inference_time = datetime.timedelta(seconds=current_time - prev_time)
        # prev_time = current_time
        # print("\t+ Yolo detect: %s" % (inference_time))

        C = []
        # select the center bbox
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            d = ((center_x - w / 2) ** 2 + (center_y - h / 2) ** 2) ** 0.5
            C.append(d)
        min_index, min_d = min(enumerate(C), key=operator.itemgetter(1))
        # print('index:%d, min_d:%.2f' %(min_index, min_d))
        filename = img_path.split("/")[-1].split(".")[0]

        n = 0
        feature = []
        big_class = -1
        small_class = -1
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            if n == min_index:
                x0 = max(int(x1 - (x2 - x1) * 0.05), 0)
                x00 = min(int(x2 + (x2 - x1) * 0.05), h)
                y0 = max(int(y1 - (y2 - y1) * 0.05), 0)
                y00 = min(int(y2 + (y2 - y1) * 0.05), w)

                pil_img = Image.open(img_path)
                pil_img = pil_img.convert('RGB')
                region = pil_img.crop((x0, y0, x00, y00))
                region.save(self.save_path + filename + '_0.jpg')

                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                input = transforms.Compose([
                    transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
                    transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
                    transforms.ToTensor(),
                    normalize,
                ])(region)
                input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)
                # switch to evaluate mode

                cls_pred = 0
                feature = []
                with torch.no_grad():
                    output_, feature = self.hrnet(input)
                    # free image
                    torch.cuda.empty_cache()

                    feature = list(feature.cpu().numpy()[0])
                    output = output_[0] + output_[1] + output_[2] + output_[3]
                    cls_pred = output.argmax(dim=1)

                if cls_pred == 8 or cls_pred == 14 or cls_pred == 19 or cls_pred == 23:
                    big_class = 1
                elif cls_pred == 5 or cls_pred == 10 or cls_pred == 12 or cls_pred == 17 or cls_pred == 21:
                    big_class = 2
                elif cls_pred == 0 or cls_pred == 4 or cls_pred == 7 or cls_pred == 11 or cls_pred == 20:
                    big_class = 3
                elif cls_pred == 1 or cls_pred == 6:
                    big_class = 4
                else:
                    big_class = 0

                small_class = cls_pred
            n = n + 1

        big = self.big_label[int(big_class)]
        small = self.small_label[int(small_class)]
        feature = np.float64(feature)
        # feature = map(np.float64, feature)
        # feature = list(feature)
        print(type(feature[0]))

        output = {'big':big, 'small':small, 'feature': feature}

        # output['big'] = big
        # output['small'] = small
        # output['feature'] = feature

        return output

    # detect clothes in the image except the shoes
    def evaluateDec(self, img_paths):
        detection, w, h = self.clothesDetection(img_paths)
        pil_img = Image.open(img_paths)
        pil_img = pil_img.convert('RGB')
        clothes_class = []
        feature = []
        n = 0
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            x0 = int(x1 - (x2 - x1) * 0.05)
            x00 = int(x2 + (x2 - x1) * 0.05)
            y0 = int(y1 - (y2 - y1) * 0.05)
            y00 = int(y2 + (y2 - y1) * 0.05)
            region = pil_img.crop((x0, y0, x00, y00))
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            input = transforms.Compose([
                transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
                transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
                transforms.ToTensor(),
                normalize,
            ])(region)
            input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)
            cls_pred = 0
            with torch.no_grad():
                output_, feature_tmp = self.hrnet(input)
                torch.cuda.empty_cache()
                feature_tmp = list(feature_tmp.cpu().numpy()[0])
                feature.append(feature_tmp)
                output = output_[0] + output_[1] + output_[2] + output_[3]
                cls_pred = output.argmax(dim=1)

            if cls_pred == 8 or cls_pred == 14 or cls_pred == 19 or cls_pred == 23:
                big_class = 1
            elif cls_pred == 5 or cls_pred == 10 or cls_pred == 12 or cls_pred == 17 or cls_pred == 21:
                big_class = 2
            elif cls_pred == 1 or cls_pred == 4 or cls_pred == 7 or cls_pred == 11 or cls_pred == 20:
                big_class = 3
            elif cls_pred == 0 or cls_pred == 6:
                big_class = 4
            else:
                big_class = 0
            small_class = cls_pred
            # save all
            clothes_type = [x1, y1, x2, y2, self.big_label[big_class], self.small_label[int(small_class)]]
            clothes_class.append(clothes_type)
        # select the max of upclothes
        C = []
        for x1, y1, x2, y2, big, small in clothes_class:
            region_h = y2 - y1
            region_w = x2 - x1
            area = region_h * region_w
            if big == "upclothes" or big == "dress":
                if small == "ldress" or small == "sdress":
                    area /= 2
                    C.append(area)
                elif small == "lskirt" or small == "sskirt":
                    C.append(-3)
                else:
                    C.append(area)
            elif big == "shoes" or big == "cap":
                C.append(-10)
            else:
                if small == "jumpsuit":
                    area /= 2
                    C.append(area)
                else:
                    C.append(-3)
        max_index, max_d = max(enumerate(C), key=operator.itemgetter(1))
        clothList = []
        featureList = []
        # select the downclothes
        max_box = clothes_class[max_index]
        if max_box[5] == "jumpsuit" or max_box[5] == "ldress" or max_box[5] == "sdress":
            clothList.append(max_box[5])
            featureList.append(feature[max_index])
        else:
            if max_box[4] == "downclothes" or (max_box[4] == "dress" and max_box[5] != "gallus"):
                clothList.append(max_box[5])
                featureList.append(feature[max_index])
            else:
                clothList.append(max_box[5])
                featureList.append(feature[max_index])
                tarx1, tary1, tarx2, tary2, tarbig, tarsmall = max_box
                tarcenterx = (tarx1 + tarx2) / 2
                # tarcentery = (tary1 + tary2) / 2
                D = []
                for x1, y1, x2, y2, big, small in clothes_class:
                    if (big == "downclothes" or big == "dress") and small != "gallus":
                        center_x = (x1 + x2) / 2
                        d = ((center_x - tarcenterx) ** 2) ** 0.5
                        D.append(d)
                    else:
                        D.append(1000)
                min_index, min_d = min(enumerate(D), key=operator.itemgetter(1))
                if min_d != 1000 and min_d < 50:
                    clothList.append(clothes_class[min_index][5])
                    featureList.append(feature[min_index])
        output = {}
        output['clothList'] = clothList
        output['featureList'] = featureList

        return output

    # 获取服装1所属的类别
    # aimFeature：服装1的512维特征
    # clusterList：服装1所属类别的两个聚类中心
    def judgeCluster(self, aimFeature, clusterList):
        cluster = -1
        distance = 0
        for i, clusterCenter in enumerate(clusterList):
            distance_tmp = 0
            for j, value in enumerate(clusterCenter):
                distance_tmp += (value - aimFeature[j]) ** 2
            if i == 0 or distance > distance_tmp:
                cluster = i
                distance = distance_tmp
        return cluster

    # 获取穿搭评价分数
    # aimDegree：关系表中两个类别交叉的数值
    # maxDegree：上衣所属类别的最大数值
    # rank：aimDegree上衣所属类别中的排序
    def getScore(self, aimDegree, maxDegree, rank):
        rank = int(rank/13)
        score = 70 + rank * 10 + (aimDegree / maxDegree) * 7
        score += random.random() * 3
        return round(score, 1)

    # 分类分割后的服装图片，返回类别以及512维特征
    def classifySegImg(self, img_path):
        pil_img = Image.open(img_path)
        pil_img = pil_img.convert('RGB')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        input = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])(pil_img)
        input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)
        # switch to evaluate mode

        cls_pred = 0
        feature = []
        with torch.no_grad():
            output_, feature = self.hrnet(input)
            # free image
            torch.cuda.empty_cache()

            feature = list(feature.cpu().numpy()[0])
            output = output_[0] + output_[1] + output_[2] + output_[3]
            cls_pred = output.argmax(dim=1)

        if cls_pred == 8 or cls_pred == 14 or cls_pred == 19 or cls_pred == 23:
            big_class = 1
        elif cls_pred == 5 or cls_pred == 10 or cls_pred == 12 or cls_pred == 17 or cls_pred == 21:
            big_class = 2
        elif cls_pred == 0 or cls_pred == 4 or cls_pred == 7 or cls_pred == 11 or cls_pred == 20:
            big_class = 3
        elif cls_pred == 1 or cls_pred == 6:
            big_class = 4
        else:
            big_class = 0

        small_class = cls_pred
        big = self.big_label[int(big_class)]
        small = self.small_label[int(small_class)]

        output = {'big': big, 'small': small, 'feature': feature}
        # output['big'] = big
        # output['small'] = small
        # output['feature'] = feature

        return output






#
#     def getShangZhuangSmallClassLable(self, input_dir):
#         from shangzhuang_attr.label_image import run_attr
#         lingxing, xiuxing = run_attr(input_dir) #属性结果-领型，袖型
#         print("lingxing")
#         print(lingxing)
#         return lingxing, xiuxing
#     def getKuZhuangSmallClassLable(self, input_dir):
#         from kuzhuang_attr.label_image import run_attr
#         kuxing = run_attr(input_dir)  # 属性结果-裤型
#         print("small_label")
#         print(kuxing)
#         return kuxing
#     def getQunZhuangSmallClassLable(self, input_dir):
#         from qunzhuang_attr.label_image import run_attr
#         lingxing1, lingxing1, qunxing1 = run_attr(input_dir)  # 属性结果-领型，袖型，裙型
#         print("lingxing")
#         print(lingxing1)
#         return lingxing1, lingxing1, qunxing1
#
# class MainHandler(tornado.web.RequestHandler):
#         def get(self):
#             picPath = self.get_argument("picPath")
#             # path = self.get_argument("path")
#             print('%s' % (picPath))
#             big_label, small_label = cloth.clothesClassify(picPath)
#             result = {
#                 "code": "0",
#                 "bigLabel": big_label,
#                 "smallLabel": small_label
#             }
#             self.write(result)
#             print("big1: " + str(big_label) + " small1: " + str(small_label) + '\n')
# #上装属性
# class MainHandlerForShangZhuang(tornado.web.RequestHandler):
#     def get(self):
#         picPath = self.get_argument("picPath")
#         # gender = self.get_argument("gender")
#         lingxing, xiuxing = cloth.getShangZhuangSmallClassLable(picPath)
#         result = {
#             "code": "0",
#             "lingxing": lingxing,
#             "xiuxing": xiuxing
#         }
#         self.write(result)
#         print(" small1: " + str(lingxing) + '\n')
# #裤装属性
# class MainHandlerForKuZhuang(tornado.web.RequestHandler):
#     def get(self):
#         picPath = self.get_argument("picPath")
#         # gender = self.get_argument("gender")
#         kuxing = cloth.getKuZhuangSmallClassLable(picPath)
#         result = {
#             "code": "0",
#             "kuxing": kuxing
#         }
#         self.write(result)
#         print(" small1: " + str(kuxing) + '\n')
# #裙装
# class MainHandlerForQunZhuang(tornado.web.RequestHandler):
#     def get(self):
#         picPath = self.get_argument("picPath")
#         # gender = self.get_argument("gender")
#         lingxing1, xiuxing1, qunxing1 = cloth.getQunZhuangSmallClassLable(picPath)
#         result = {
#             "code": "0",
#             "lingxing": lingxing1,
#             "xiuxing": xiuxing1,
#             "qunxing": qunxing1
#         }
#         self.write(result)
#         print(" lingxing: " + str(lingxing1) + '\n')
# #图片切割
# class MainHandler1(tornado.web.RequestHandler):
#     def get(self):
#         picPath = self.get_argument("picPath")
#         run_seg('./image/' + picPath)
#         result = {
#             "code": "0",
#         }
#         self.write(result)
#         print(picPath)
# #获取穿搭评价的服装列表
# class MainHandlerGetEvaluateWearList(tornado.web.RequestHandler):
#     def get(self):
#         picPath = self.get_argument("picPath")
#         clothList = cloth.clothesEvaluate(os.path.pardir+'/uploadFiles/myfile/'+picPath)
#         result = {
#             "code": "0",
#             "clothList": clothList,
#         }
#         self.write(result)
# #获取搭配列表
# class MainHandlerGetMatchClothesList(tornado.web.RequestHandler):
#     def get(self):
#         sex = self.get_argument("sex")
#         style = self.get_argument("style")
#         clothid = self.get_argument("clothid")
#         bigClass = self.get_argument("bigclass")
#         smallClass = self.get_argument("smallclass")
#         if sex is 'female':
#             if style2 is 'richang':
#                 style = 'suixing'
#             elif style2 is 'yuehui':
#                 style = 'keai'
#             elif style2 is 'zhichang':
#                 style = 'zhichang'
#             else:
#                 style = 'youya'
#         if sex is 'male':
#             if style2 is 'richang':
#                 style = 'suixing'
#             elif style2 is 'yuehui':
#                 style = 'shishang'
#             elif style2 is 'zhichang':
#                 style = 'zhichang'
#             else:
#                 style = 'qichang'
#         if sex is '0':
#            sex = "female"
#         else:
#            sex = "male"
#         if style is '1':
#            style = "richang"
#         elif style is '2':
#            style = "yuehui"
#         elif style is '3':
#            style = "zhichang"
#         elif style is '4':
#            style = "youya"
#         else:
#            style = "youya"
#         if clothid is 'S':
#             clothid = None
#         if bigClass is 'S':
#             bigClass = None
#         if smallClass is 'S':
#             smallClass = None
#         #totalNum, clothList, clothList1 = Costumematching("female","richang",None,None,None)
#         totalNum, clothList, clothList1 = Costumematching(str(sex), str(style), str(clothid), str(bigClass), str(smallClass))
#         print(totalNum)
#         print(clothList)
#         print(clothList1)
#         result = {
#             "code": "0",
#             "totalNum": str(totalNum),
#             "clothList": clothList,
#             "clothList1": clothList1,
#         }
#         self.write(result)
# def make_app():
#     return tornado.web.Application([
#         (r"/clothesClassify", MainHandler),
#         (r"/getKuZhuangSmallClassLable", MainHandlerForKuZhuang),
#         (r"/getShangZhuangSmallClassLable", MainHandlerForShangZhuang),
#         (r"/getQunZhuangSmallClassLable", MainHandlerForQunZhuang),
#         (r"/getClothesClass", MainHandler1),
#         (r"/getEvaluateWearList", MainHandlerGetEvaluateWearList),
#         (r"/getMatchClothesList", MainHandlerGetMatchClothesList),
#     ])
if __name__ == '__main__':
    cloth = HisClothesClassify("../saveFile/")
    cloth.initModel()
    prev_time = time.time()

    output1 = cloth.clothesClassify("../saveFile/1.jpg")
    print(output1)

    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ clothesClassify: %s" % (inference_time))

    output2 = cloth.evaluateDec("../saveFile/0.png")
    print(output2)

    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ evaluateDec: %s" % (inference_time))

    num = cloth.judgeCluster(output1['feature'], output2['featureList'])
    print(num)
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ judgeCluster: %s" % (inference_time))

    score = cloth.getScore(13, 99, 35)
    print(score)
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ judgeCluster: %s" % (inference_time))

    output3 = cloth.classifySegImg("../saveFile/0.png")
    print(output3)
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ classifySegImg: %s" % (inference_time))
    # big_label, small_label = cloth.clothesClassify("6.jpg")
    # print("big: " + str(big_label) + " small: " + str(small_label) + '\n')
    # lingxing, xiuxing = cloth.getShangZhuangSmallClassLable("098.jpg")
    # print(" small1: " + str(lingxing) + '\n')
    # small_class_lable = cloth.getKuZhuangSmallClassLable("878.jpg")
    # print(" small1: " + str(small_class_lable) + '\n')
    # lingxing1, xiuxing1, qunxing1 = cloth.getQunZhuangSmallClassLable("6.jpg")
    # print(" small1: " + str(lingxing1) + '\n') scoat1.jpg  Costumematching("female","richang","sweater1","upclothes","scoat")
    # Costumematching("female","youya","sweater1","upclothes","scoat")
    # app = tornado.httpserver.HTTPServer(make_app())
    # app.listen(8888)
    # tornado.ioloop.IOLoop.current().start()
    # print("启动成功")
