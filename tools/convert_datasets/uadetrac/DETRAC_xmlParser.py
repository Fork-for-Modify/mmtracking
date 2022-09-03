
# ================================
# convert UA-DETRAC xml annotation files (video-wise) to VOC style xml annotation files (image-wise)
# Modified: Zhihong Zhang
# From: https://github.com/w5688414/datasets-preprocessing-for-object-detection
# ================================

import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import os
import cv2
import time


def ConvertVOCXml(file_path="", file_name=""):
    tree = ET.parse(file_name)
    root = tree.getroot()
    # print(root.tag)

    num = 0  # 计数
    #读xml操作

    frame_lists = []
    output_file_name = ""
    for child in root:

        if(child.tag == "frame"):
            # 创建dom文档
            doc = Document()
            # 创建根节点
            annotation = doc.createElement('annotation')
            # 根节点插入dom树
            doc.appendChild(annotation)

            #print(child.tag, child.attrib["num"])
            pic_id = child.attrib["num"].zfill(5)
            #print(pic_id)
            output_file_name = root.attrib["name"]+"/img"+pic_id+".xml"
            #  print(output_file_name)

            folder = doc.createElement("folder")
            folder.appendChild(doc.createTextNode(
                output_file_name.split(os.sep)[0]))
            annotation.appendChild(folder)

            filename = doc.createElement("filename")
            pic_name = "img"+pic_id+".jpg"
            filename.appendChild(doc.createTextNode(pic_name))
            annotation.appendChild(filename)

            sizeimage = doc.createElement("size")
            imagewidth = doc.createElement("width")
            imageheight = doc.createElement("height")
            imagedepth = doc.createElement("depth")

            imagewidth.appendChild(doc.createTextNode("960"))
            imageheight.appendChild(doc.createTextNode("540"))
            imagedepth.appendChild(doc.createTextNode("3"))

            sizeimage.appendChild(imagedepth)
            sizeimage.appendChild(imagewidth)
            sizeimage.appendChild(imageheight)
            annotation.appendChild(sizeimage)

            target_list = child.getchildren()[0]  # 获取target_list
            #print(target_list.tag)
            object = None
            for target in target_list:
                if(target.tag == "target"):
                    #print(target.tag)
                    object = doc.createElement('object')
                    trackid = doc.createElement("trackid")
                    bndbox = doc.createElement("bndbox")

                    trackid.appendChild(
                        doc.createTextNode(target.attrib['id']))

                    for target_child in target:
                        if(target_child.tag == "box"):
                            xmin = doc.createElement("xmin")
                            ymin = doc.createElement("ymin")
                            xmax = doc.createElement("xmax")
                            ymax = doc.createElement("ymax")
                            xmin_value = int(
                                float(target_child.attrib["left"]))
                            ymin_value = int(float(target_child.attrib["top"]))
                            box_width_value = int(
                                float(target_child.attrib["width"]))
                            box_height_value = int(
                                float(target_child.attrib["height"]))
                            xmin.appendChild(
                                doc.createTextNode(str(xmin_value)))
                            ymin.appendChild(
                                doc.createTextNode(str(ymin_value)))
                            if(xmin_value+box_width_value > 960):
                                xmax.appendChild(doc.createTextNode(str(960)))
                            else:
                                xmax.appendChild(doc.createTextNode(
                                    str(xmin_value+box_width_value)))
                            if(ymin_value+box_height_value > 540):
                                ymax.appendChild(doc.createTextNode(str(540)))
                            else:
                                ymax.appendChild(doc.createTextNode(
                                    str(ymin_value+box_height_value)))

                        if(target_child.tag == "attribute"):
                            name = doc.createElement('name')
                            speed = doc.createElement('speed')
                            truncation_ratio = doc.createElement(
                                'truncation_ratio')
                            difficult = doc.createElement('difficult')

                            name.appendChild(
                                doc.createTextNode(target_child.attrib['vehicle_type']))
                            speed.appendChild(
                                doc.createTextNode(target_child.attrib['speed']))
                            truncation_ratio.appendChild(
                                doc.createTextNode(target_child.attrib['truncation_ratio']))
                            difficult.appendChild(
                                doc.createTextNode("-1"))

                            object.appendChild(name)
                            object.appendChild(trackid)
                            object.appendChild(speed)
                            object.appendChild(truncation_ratio)
                            object.appendChild(difficult)

                    bndbox.appendChild(xmin)
                    bndbox.appendChild(ymin)
                    bndbox.appendChild(xmax)
                    bndbox.appendChild(ymax)
                    object.appendChild(bndbox)
                    annotation.appendChild(object)

            file_path_out = os.path.join(file_path, output_file_name)
            if not os.path.exists(file_path_out.rsplit('/', 1)[0]):
                os.makedirs(file_path_out.rsplit('/', 1)[0])
            f = open(file_path_out, 'w')
            f.write(doc.toprettyxml(indent=' ' * 4))
            f.close()
            num = num+1
    return num


'''
画方框
'''


def bboxes_draw_on_img(img, bbox, color=[255, 0, 0], thickness=2):

    # Draw bounding box...
    print(bbox)
    p1 = (int(float(bbox["xmin"])), int(float(bbox["ymin"])))
    p2 = (int(float(bbox["xmax"])), int(float(bbox["ymax"])))
    cv2.rectangle(img, p1, p2, color, thickness)


def visualization_image(image_name, xml_file_name):
    tree = ET.parse(xml_file_name)
    root = tree.getroot()

    object_lists = []
    for child in root:
        if(child.tag == "folder"):
            print(child.tag, child.text)
        elif (child.tag == "filename"):
            print(child.tag, child.text)
        elif (child.tag == "size"):  # 解析size
            for size_child in child:
                if(size_child.tag == "width"):
                    print(size_child.tag, size_child.text)
                elif (size_child.tag == "height"):
                    print(size_child.tag, size_child.text)
                elif (size_child.tag == "depth"):
                    print(size_child.tag, size_child.text)
        elif (child.tag == "object"):  # 解析object
            singleObject = {}
            for object_child in child:
                if (object_child.tag == "name"):
                    # print(object_child.tag,object_child.text)
                    singleObject["name"] = object_child.text
                elif (object_child.tag == "bndbox"):
                    for bndbox_child in object_child:
                        if (bndbox_child.tag == "xmin"):
                            singleObject["xmin"] = bndbox_child.text
                            # print(bndbox_child.tag, bndbox_child.text)
                        elif (bndbox_child.tag == "ymin"):
                            # print(bndbox_child.tag, bndbox_child.text)
                            singleObject["ymin"] = bndbox_child.text
                        elif (bndbox_child.tag == "xmax"):
                            singleObject["xmax"] = bndbox_child.text
                        elif (bndbox_child.tag == "ymax"):
                            singleObject["ymax"] = bndbox_child.text
            object_length = len(singleObject)
            if(object_length > 0):
                object_lists.append(singleObject)
    img = cv2.imread(image_name)
    for object_coordinate in object_lists:
        bboxes_draw_on_img(img, object_coordinate)
    # cv2.imshow("capture", img)
    cv2.imwrite("./" + image_name.rsplit('/', 1)[1], img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if (__name__ == "__main__"):
    # path setting
    basePath = "/hdd/0/zzh/dataset/UA_DETRAC/original_style/DETRAC-Test-Annotations-XML/"
    saveBasePath = "/hdd/0/zzh/dataset/UA_DETRAC/coco_style/annotations_xml/VID/test/"
    imgPath = "/hdd/0/zzh/dataset/UA_DETRAC/coco_style/Data/VID/test/"

    # convert
    totalxml = os.listdir(basePath)
    total_num = 0
    flag = False
    print("正在转换")

    if os.path.exists(saveBasePath) == False:  # 判断文件夹是否存在
        os.makedirs(saveBasePath)

    # Start time
    start = time.time()
    log = open("./DETRAC_xmlParser_log.txt", "w")  # 分析日志，进行排错
    for xml in totalxml:
        file_name = os.path.join(basePath, xml)
        print(file_name)
        num = ConvertVOCXml(file_path=saveBasePath, file_name=file_name)
        print(num)
        total_num = total_num+num
        log.write(file_name+" "+str(num)+"\n")
    # End time
    end = time.time()
    seconds = end-start
    print("Time taken : {0} seconds".format(seconds))
    print(total_num)
    log.write(str(total_num)+"\n")

    # visualization to check
    # visualization_image(imgPath+"MVI_39031/img00396.jpg",
    #                     saveBasePath+"MVI_39031/00396.xml")
