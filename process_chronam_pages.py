# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # This Newspaper Navigator Dataset Pipeline
# 
# By Benjamin Charles Germain Lee (2020 Library of Congress Innovator-in-Residence)
# 
# Using the manifests saved in the repo, this notebook iterates over each manifest (corresponding to a Chronicling America batch and performs the following.  *Note that if you would like to use an updated version of the manifests, please see the repo "chronam-get-images" (https://github.com/bcglee/chronam-get-images) for how to generate the manifests.*
# 
# 1. Systematically downloads the images and METS/ALTO OCR (XML) from the proper S3 buckets. This step allows for CPU multiprocessing; adjust N_CPU_PROCESSES as desired in main(). *Note: if you're experimenting with this pipeline and can't configure access to the Chronicling America S3 buckets, you can use the repo chronam-get-images (https://github.com/bcglee/chronam-get-images) to pull down the JPG and XML files of desired pages; then, move those files to '../chronam_files').*
# 2. Generates and saves predictions for each JPG image. This step utilizes model weights generated using the notebook 'train_model.ipynb'. This step uses all available GPUs.
# 3. Adds captions from the METS/ALTO OCR in the XML for each image as metadata. This step is performed by identifying text within each predicted bounding box. This step allows for CPU multiprocessing.
# 4. Crops all of the predicted visual content and saves the cropped images. This step allows for CPU multiprocessing.
# 5. Generates embeddings for the predicted visual content for each image and adds the embeddings to the metadata. Currently, img2vec is being utilized for this (https://github.com/christiansafka/img2vec). This step uses all available GPUs.
# 6. Packages the files according to the Chronicling America file structure and sends the files to an S3 bucket. The downloaded files and generated metadata are all then deleted to free up space.
# 
# The pipeline is driven by main(), which is the last cell in the notebook. The functions used in main() appear below with short descriptions. Though this workflow differs from most Jupyter notebooks, the notebook format is being used here for the ease of annotating code.
# 
# NOTE: if you would like to run this code, you MUST save the notebook as a Python file using the command "jupyter nbconvert --to script process_chronam_pages.ipynb" and run the Python script using "python process_chronam_pages.py".  This is necessary because the notebook is unable to handle multiprocessing.
# %% [markdown]
# # The next two cells include the code for the systematic download of Chroncling America images from the S3 buckets.
# 
# This first cell handles imports and initial settings for pulling down the files:

# %%
import boto3
import botocore
import s3fs
import glob
import sys
import os
import time
from PIL import Image
import io
import math
import datetime
import subprocess
import collections
import math

# %% [markdown]
# # This second cell handles the function for file retrieval for a specified manifest and destination directory.  
# 
# Note that we test first to see if a JPG exists in 'ndnp-jpeg-surrogates' and if not, we then grab the JP2 from 'ndnp-batches' (converting the JP2 to JPG requires overhead). The XML file is then downloaded.

# %%
# function that retrieves .jpg and .xml files for each filepath in manifest
def retrieve_files(packet):
    
    # sets boto3 to run with s3
    s3 = boto3.resource('s3')
    
    # use s3fs file system for checking file existence
    s3fs_filye_sys = s3fs.S3FileSystem()        

    # creates dict for storing widths/heights of images
    im_size_dict = {}  
    
    # grab directory to CD into first (it is the firs entry in the array)
    dir_path = packet[0]
    os.chdir(dir_path)
    
    # grabs page_filepaths from the data packet
    page_filepaths = packet[1]
        
    # iterate through each filepath and download
    for page_filepath in page_filepaths:
                
        # sets filepath for download destination (note: file is .jp2, so we need to replace suffixes below)
        local_filepath = page_filepath.replace('/', '_')

        # see: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/migrations3.html
        # see also:  https://www.edureka.co/community/17558/python-aws-boto3-how-do-i-read-files-from-s3-bucket
        try:

            # if the JPEG exists in ndnp-jpeg-surrogates, pull down from that S3 bucket
            if s3fs_filye_sys.exists('ndnp-jpeg-surrogates/' + page_filepath.replace(".jp2", ".jpg")):
                obj = s3.Object('ndnp-jpeg-surrogates', page_filepath.replace(".jp2", ".jpg"))
                body = obj.get()['Body'].read()
                im = Image.open(io.BytesIO(body))
                im.resize((math.floor(im.width/6), math.floor(im.height/6)), resample=0).save(local_filepath.replace(".jp2", ".jpg"))
                im_size_dict[local_filepath.replace(".jp2", ".jpg")] = (im.width, im.height)

            # if the JPEG doesn't exist, pull down the JPEG-2000 from 'ndnp-batches' S3 bucket and convert to JPG
            else:
                s3.Bucket('ndnp-batches').download_file(page_filepath, local_filepath)
                subprocess.call('gm convert ' + local_filepath + ' ' + local_filepath.replace(".jp2", ".jpg"), shell=True)
                subprocess.call('rm ' + local_filepath, shell=True)

                if not os.path.exists(local_filepath.replace(".jp2", ".jpg")):
                    continue

                im = Image.open(local_filepath.replace(".jp2", ".jpg"))
                im.resize((math.floor(im.width/6), math.floor(im.height/6)), resample=0).save(local_filepath.replace(".jp2", ".jpg"))
                im_size_dict[local_filepath.replace(".jp2", ".jpg")] = (im.width, im.height)

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                print("Error in downloading JP2.")
                raise
                
        try:
            s3.Bucket('ndnp-batches').download_file(page_filepath.replace(".jp2", ".xml"), local_filepath.replace(".jp2", ".xml"))
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                print("Error in downloading XML.")
                raise
        
    return im_size_dict

# %% [markdown]
# # The next two cells load the finetuned model and define the function for performing predictions on the images.

# %%
# import some common libraries
import cv2
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# import deep learning imports
import detectron2
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


# %%
def generate_predictions(zipped):
            
    # unzips packed information for process to perform predictions
    
    S3_SAVE_DIR = zipped[0]
    OUTPUT_SAVE_DIR = zipped[1]
    dir_name = zipped[2]
    INFERENCE_BATCH_SIZE = zipped[3]
    filepaths = zipped[4]
    ID = zipped[5]

    with torch.cuda.device(ID):

        # navigates to correct directory (process is spawned in /notebooks)

        os.chdir(S3_SAVE_DIR + dir_name)

        # sets up model for process

        setup_logger()
        cfg = get_cfg()
        cfg.merge_from_file("../../..//detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

        # sets prediction score threshold - this is commented out and defaults to 0.05 in Detectron2
        # if you would like to adjust the threshold, uncomment and set to the desired value in [0, 1)
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        # sets number of object classes to 7
        # ("Illustration/Photograph", "Photograph", "Comics/Cartoon", "Editorial Cartoon", "Map", "Headline", "Ad")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

        # build model
        model = build_model(cfg)

        # see:  https://github.com/facebookresearch/detectron2/issues/282 
        # (must load weights this way if using model)
        DetectionCheckpointer(model).load("../../model_weights/model_final.pth")
        model.train(False) 

        # construct batches
        batches = chunk(filepaths, math.ceil(len(filepaths)/INFERENCE_BATCH_SIZE))

        # iterate through images
        for batch in batches:

            # sets up inputs by loading in all files in batch
            inputs = []

            # stores image dimensions
            dimensions = []

            # iterate through files in batch
            for file in batch:

                # read in image
                image = cv2.imread(file)

                # store image dimensions
                height, width, _ = image.shape
                dimensions.append([width, height])

                # perform inference on batch
                image = np.transpose(image,(2,0,1))
                # see https://github.com/facebookresearch/detectron2/issues/282 for in-depth description of why 
                # image is loaded in this way
                image_tensor = torch.from_numpy(image)
                inputs.append({"image": image_tensor})

            # performs inference
            outputs = model(inputs)

            # saves predictions
            predictions = {}

            # iterate over images in batch and save predictions to JSON
            for i in range(0, len(batch)):

                # saves filepath in format of ChronAm file structure
                predictions["filepath"] = dir_name + "data/" + batch[i].split("data_")[1].replace(dir_name, '').replace('_', '/').replace('.jpg', '.jp2')

                # parses metadata from filepath
                date_str = predictions["filepath"].split('/')[-2]
                predictions["batch"] = dir_name[:-1]
                predictions["lccn"] = predictions["filepath"].split('/')[-4]
                predictions["pub_date"] = str(datetime.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])))
                predictions["edition_seq_num"] = int(date_str[8:10])
                if "a" in predictions["filepath"].split('/')[-1][:-4] or "b" in predictions["filepath"].split('/')[-1][:-4]:
                    continue
                predictions["page_seq_num"] = int(predictions["filepath"].split('/')[-1][:-4])
                
                # saves predictions
                # we first normalize the bounding box coordinates
                boxes = outputs[i]["instances"].get_fields()["pred_boxes"].to("cpu").tensor.tolist()
                normalized_boxes = []
                width = dimensions[i][0]
                height = dimensions[i][1]

                for box in boxes:
                    normalized_box = (box[0]/float(width), box[1]/float(height), box[2]/float(width), box[3]/float(height))
                    normalized_boxes.append(normalized_box)

                # saves additional outputs of predictions
                predictions["boxes"] = normalized_boxes
                predictions["scores"] = outputs[i]["instances"].get_fields()["scores"].to("cpu").tolist()
                predictions["pred_classes"] = outputs[i]["instances"].get_fields()["pred_classes"].to("cpu").tolist()

                with open(OUTPUT_SAVE_DIR + dir_name + batch[i].replace('.jpg','.json'), "w") as fp:
                    json.dump(predictions, fp)

# %% [markdown]
# # The next two cells define functions for extracting the OCR within each predicted box.
# 
# 1. The first cell defines the function for returning the proper OCR for a specific page.
# 2. The second cell defines the function for iterating over the JSON files containing the predictions.

# %%
# import xml.etree.cElementTree as ET
# etree
from lxml import etree as ET
from xml.etree.ElementTree import ElementTree

# tolerance around box for testing whether OCR falls within bounds
WIDTH_TOLERANCE = 0.000
HEIGHT_TOLERANCE = 0.000

# given a file path and a list of bounding boxes, this function traverses the associated XML
# and returns the OCR within each bounding box
def retrieve_ocr_for_file(xml_filepath, true_img_filepath, page_width_pix, page_height_pix, bounding_boxes, predicted_classes):

    # creates empty nested list fo storing OCR in each box
    ocr = [ [] for i in range(len(bounding_boxes)) ]

    # sets tree and root based on filepath
    parser = ET.XMLParser()
    tree = ET.parse(xml_filepath, parser)
    root = tree.getroot()
    
    # sets tag prefix (everywhere)
    prefix = root.tag.split('}')[0] + '}'

    # traverses to layout and then the page and then the print space
    layout = root.find(prefix + 'Layout')
    page = layout.find(prefix + 'Page')
    print_space = page.find(prefix + 'PrintSpace')
    
    if print_space is None:
        return ocr

    text_boxes =  [textblock for textblock in print_space.iterchildren(prefix + "TextBlock")]
    
    # gets page height and page width in inch1200 units
    page_width_inch = int(page.attrib['WIDTH'])
    page_height_inch = int(page.attrib['HEIGHT'])

    # sets conversion to normalized coordinates for comparison between METS/ALTO and predicted boxes
    W_CONVERSION = 1./float(page_width_inch)
    H_CONVERSION = 1./float(page_height_inch)

    if page_width_inch == 0 or page_height_inch == 0:
        return ocr

    # we now iterate over each bounding box
    for i in range(0, len(bounding_boxes)):

        bounding_box = bounding_boxes[i]
        predicted_class = predicted_classes[i]

        # we then iterate over each text box
        for text_box in text_boxes:
                        
            box_w1 = int(float(text_box.attrib["HPOS"]))
            box_h1 = int(float(text_box.attrib["VPOS"]))
            box_w2 = box_w1 + int(float(text_box.attrib["WIDTH"]))
            box_h2 = box_h1 + int(float(text_box.attrib["HEIGHT"]))
            
            # if the text box and bounding box do not intersect, we skip (as no text will overlap in smaller units)
            if box_w2*W_CONVERSION < bounding_box[0] and box_h2*H_CONVERSION < bounding_box[1]:
                continue
            if box_w1*W_CONVERSION > bounding_box[0] + bounding_box[2] and box_h2*H_CONVERSION < bounding_box[1]:
                continue
            if box_w2*W_CONVERSION < bounding_box[0] and box_h1*H_CONVERSION > bounding_box[1] + bounding_box[3]:
                continue
            if box_w1*W_CONVERSION > bounding_box[0] + bounding_box[2] and box_h1*H_CONVERSION > bounding_box[1] + bounding_box[3]:
                continue
                
            # we then iterate over the text lines in each box
            for text_line in text_box.iterchildren(prefix + 'TextLine'):
                
                line_w1 = int(float(text_box.attrib["HPOS"]))
                line_h1 = int(float(text_box.attrib["VPOS"]))
                line_w2 = line_w1 + int(float(text_box.attrib["WIDTH"]))
                line_h2 = line_h1 + int(float(text_box.attrib["HEIGHT"]))

                # if the text box and bounding box do not intersect, we skip (as no text will overlap in smaller units)
                if line_w2*W_CONVERSION < bounding_box[0] and line_h2*H_CONVERSION < bounding_box[1]:
                    continue
                if line_w1*W_CONVERSION > bounding_box[0] + bounding_box[2] and line_h2*H_CONVERSION < bounding_box[1]:
                    continue
                if line_w2*W_CONVERSION < bounding_box[0] and line_h1*H_CONVERSION > bounding_box[1] + bounding_box[3]:
                    continue
                if line_w1*W_CONVERSION > bounding_box[0] + bounding_box[2] and line_h1*H_CONVERSION > bounding_box[1] + bounding_box[3]:
                    continue
                
                # we now iterate over every string in each line (each string is separated by whitespace)
                for string in text_line.iterchildren(prefix + 'String'):
            
                    w1 = int(float(string.attrib["HPOS"]))
                    h1 = int(float(string.attrib["VPOS"]))
                    w2 = w1 + int(float(string.attrib["WIDTH"]))
                    h2 = h1 + int(float(string.attrib["HEIGHT"]))

                    # checks if the text appears within the bounding box & extra tolerance for words that are clipped
                    if w1*W_CONVERSION > bounding_box[0] - WIDTH_TOLERANCE:
                        if w2*W_CONVERSION < bounding_box[2] + WIDTH_TOLERANCE:
                            if h1*H_CONVERSION > bounding_box[1] - HEIGHT_TOLERANCE:
                                if h2*H_CONVERSION < bounding_box[3] + HEIGHT_TOLERANCE:

                                    # appends text content to list
                                    ocr[i].append(string.attrib["CONTENT"])

    return ocr


# %%
def retrieve_ocr(packet):

    # grab contents of packet, CD into correct directory
    dir_name = packet[1]
    os.chdir(packet[0] + dir_name)
    json_info = packet[2]

    # we now iterate through all of the predictions JSON files
    for json_entry in json_info:
        
        # unpacks the input from Pool
        json_filepath = json_entry[0]
        im_width = json_entry[1]
        im_height = json_entry[2]
        
        # loads the JSON
        with open(json_filepath) as f:
            predictions = json.load(f)
        
        # pulls off relevant data fields from the JSON
        original_img_filepath = predictions['filepath']
        boxes = predictions['boxes']
        scores = predictions['scores']
        classes = predictions['pred_classes']

        # sets the number of predicted bounding boxes
        n_pred = len(scores)

        # we now find the XML and JPG files corresponding to this predictions JSON
        xml_filepath = S3_SAVE_DIR + dir_name + json_filepath.replace('.json', '.xml')
        jpg_filepath = S3_SAVE_DIR + dir_name + json_filepath.replace('.json', '.jpg')

        # stores list of OCR
        ocr = []

        # we only try to retrieve the OCR if there is one or more predicted box
        if n_pred > 0:
            ocr = retrieve_ocr_for_file(xml_filepath, jpg_filepath, im_width, im_height, boxes, classes)

        # adds the ocr field to the JSON metadata for the page
        predictions['ocr'] = ocr

        # we save the updated JSON
        with open(json_filepath, 'w') as f:
            json.dump(predictions, f)

# %% [markdown]
# # This cell defines a function for cropping all of the predicted visual content:

# %%
def crop(packet):
            
    OUTPUT_SAVE_DIR = packet[0]
    S3_SAVE_DIR = packet[1]
    dir_name = packet[2]
    json_filepaths = packet[3]
    
    os.chdir(OUTPUT_SAVE_DIR+dir_name)

    for json_filepath in json_filepaths:
        
        # we load the JSON
        with open(json_filepath) as f:
            predictions = json.load(f)
          
        # load in boxes
        boxes = predictions['boxes']
        scores = predictions['scores']
        classes = predictions['pred_classes']
        
        # grab filepath of image
        jpg_filepath = S3_SAVE_DIR + dir_name + json_filepath.replace('.json', '.jpg')

        # open image
        im = Image.open(jpg_filepath)
        
        # empty list for storing embeddings
        img_embeddings = []
        
        # empty list or storing filepaths of extracted visual content
        content_filepaths = []

        # iterate through boxes, crop, and send to embedding
        for i in range(0, len(boxes)):
            box = boxes[i]
            pred_class = classes[i]
            score = scores[i]
            
            # if it's a headline or the confidence score is less than 0.5, we skip the cropping
            if pred_class == 5:
                img_embeddings.append([])
                content_filepaths.append([])
                continue
                
            # crop image according to box (converted from normalized coordinates to image coordinates)
            cropped = im.crop((box[0]*im.width, box[1]*im.height, box[2]*im.width, box[3]*im.height)).convert('RGB')
            # save cropped image to output directory
            cropped_filepath = json_filepath.replace(".json", "_" + str(i).zfill(3) + "_" + str(pred_class) + "_" + str(int(math.floor(100*score))).zfill(2) + ".jpg")
            cropped.save(cropped_filepath)
            new_filepath = dir_name + "data/" + cropped_filepath.split("data_")[1].replace(dir_name, '').replace('_', '/') 
            new_filepath = new_filepath[:new_filepath.rfind("/")] + "_" + new_filepath[new_filepath.rfind("/")+1:]
            new_filepath = new_filepath[:new_filepath.rfind("/")] + "_" + new_filepath[new_filepath.rfind("/")+1:]
            content_filepaths.append(new_filepath)

        # add filepaths of extracted visual content to output
        predictions['visual_content_filepaths'] = content_filepaths
        
        # we save the updated JSON
        with open(json_filepath, 'w') as f:
            json.dump(predictions, f)
    

# %% [markdown]
# # This cell defines a function for generating embeddings of each predicted box:

# %%
from img2vec_pytorch import Img2Vec

def generate_embeddings(zipped):
    
    # unzips packed information for process to perform predictions
    
    OUTPUT_SAVE_DIR = zipped[0]
    S3_SAVE_DIR = zipped[1]
    dir_name = zipped[2]
    json_filepaths = zipped[3]
    ID = zipped[4]

    with torch.cuda.device(ID):

        # load in img2vec
        # we choose resnet embeddings
        img2vec_resnet_50 = Img2Vec(cuda=True, model='resnet-50') 
        img2vec_resnet_18 = Img2Vec(cuda=True, model='resnet-18') 
    
        # iterate through the JSON files
        for json_filepath in json_filepaths:
            
            # we load the JSON
            with open(json_filepath) as f:
                predictions = json.load(f)

            # load in boxes
            boxes = predictions['boxes']
            scores = predictions['scores']
            classes = predictions['pred_classes']
            cropped_filepaths = predictions['visual_content_filepaths']

            # grab filepath of image
            jpg_filepath = S3_SAVE_DIR + dir_name + json_filepath.replace('.json', '.jpg')

            # empty list for storing embeddings
            resnet_50_embeddings = []
            resnet_18_embeddings = []

            # iterate through boxes, crop, and send to embedding
            for i in range(0, len(boxes)):

                box = boxes[i]
                pred_class = classes[i]
                score = scores[i]
                
                # if it's a headline or confidence score is less than 0.5, we skip the embedding generation
                if pred_class == 5 or score < 0.5:
                    resnet_50_embeddings.append([])
                    resnet_18_embeddings.append([])
                    continue

                cropped_filepath = cropped_filepaths[i]
                # reformat to use flat file directory
                cropped_filepath = cropped_filepath.replace("/", "_")
                
                # open cropped image
                im = Image.open(cropped_filepath).convert('RGB')
                # generate embedding using img2vec
                embedding_resnet_50 = img2vec_resnet_50.get_vec(im, tensor=False)
                embedding_resnet_18 = img2vec_resnet_18.get_vec(im, tensor=False)
                # add to list (render embedding numpy array as list to enable JSON serialization)
                resnet_50_embeddings.append(embedding_resnet_50.tolist())
                resnet_18_embeddings.append(embedding_resnet_18.tolist())
                
            embeddings_json = {}
            embeddings_json['filepath'] = predictions['filepath']
            embeddings_json['visual_content_filepaths'] = predictions['visual_content_filepaths']
            # add embeddings to output
            embeddings_json['resnet_50_embeddings'] = resnet_50_embeddings
            embeddings_json['resnet_18_embeddings'] = resnet_18_embeddings

            # we save the updated JSON
            with open(json_filepath[:-5] + "_embeddings.json", 'w') as f:
                json.dump(embeddings_json, f)

# %% [markdown]
# # The cell below defines a function for chunking data for multiprocessing:

# %%
# function that splits a list into n chunks for multiprocessing
def chunk(file_list, n_chunks):
    
    # make chunks of files to be distributed across processes
    chunks = []
    chunk_size = math.ceil(float(len(file_list))/n_chunks)
    for i in range(0, n_chunks-1):
        chunks.append(file_list[i*chunk_size:(i+1)*chunk_size])
    chunks.append(file_list[(n_chunks-1)*chunk_size:])
    
    return chunks

# %% [markdown]
# # The cell below defines a function for validating that JP2 and XML filepaths derived from the manifests exist:

# %%
# function that determines whether the JP2 and XML files exist for specified files
def files_exist(filepaths):
    
    no_jp2 = []
    no_xml = []
    good_filepaths = []
    s3 = s3fs.S3FileSystem()
    for filepath in filepaths:
        if not s3.exists('ndnp-batches/' + filepath):
            no_jp2.append(filepath)
            continue
        if not s3.exists('ndnp-batches/' + filepath.replace(".jp2", ".xml")):
            no_xml.append(filepath.replace(".jp2", ".xml"))
            continue
        good_filepaths.append(filepath)

    return [good_filepaths, no_jp2, no_xml]

# %% [markdown]
# # The cell below defines a function for uploading files to an S3 bucket:

# %%
def upload_files(filepaths):
    
    # cd into the 'save' directory 
    os.chdir(OUTPUT_SAVE_DIR)
    
    # connects to boto3
    s3 = boto3.resource('s3')
    
    # uploads files to mirror the ChronAm structure
    for filepath in filepaths:
        
        if '_stats.json' in filepath or '.txt' in filepath: 
            s3.Bucket("S3-BUCKET-HERE").upload_file(filepath, "chronam_processed/"+filepath)
            continue
            
        if '.json' in filepath:
            dir_name = filepath.split("/")[0] + "/"
            new_filepath = dir_name + "data/" + filepath.split("data_")[1].replace(dir_name, '').replace('_', '/')
            s3.Bucket("S3-BUCKET-HERE").upload_file(filepath, "chronam_processed/"+new_filepath)
            continue

        if '.jpg' in filepath:
            dir_name = filepath.split("/")[0] + "/"
            new_filepath = dir_name + "data/" + filepath.split("data_")[1].replace(dir_name, '').replace('_', '/') 
            new_filepath = new_filepath[:new_filepath.rfind("/")] + "_" + new_filepath[new_filepath.rfind("/")+1:]
            new_filepath = new_filepath[:new_filepath.rfind("/")] + "_" + new_filepath[new_filepath.rfind("/")+1:]
            s3.Bucket("S3-BUCKET-HERE").upload_file(filepath, "chronam_processed/"+new_filepath)
            continue

# %% [markdown]
# # The cell below contains main(), the driver of the pipeline.

# %%
from multiprocessing import Pool, get_context, Process, set_start_method
from collections import ChainMap
import shutil
import time

# need main for setting multiprocessing start method to spawn
if __name__ == '__main__':
    
    # sets directory location where the notebook is
    NOTEBOOK_DIR = os.getcwd()
    os.chdir('../')
    # sets destination for saving downloaded S3 files
    S3_SAVE_DIR = os.getcwd() + '/chronam_files/'
    # sets destination for output files, containing new metadata
    OUTPUT_SAVE_DIR = os.getcwd() + '/chronam_output/'
    os.chdir('notebooks/')

    # construct the directories
    if not os.path.isdir(S3_SAVE_DIR):
        os.mkdir(S3_SAVE_DIR)
    if not os.path.isdir(OUTPUT_SAVE_DIR):
        os.mkdir(OUTPUT_SAVE_DIR)

    # sets batch size for GPU inference; using 1 to prevent any memory overflow edge case in pipeline
    INFERENCE_BATCH_SIZE = 1

    # sets number of processes (be careful based on number of available cores)
    N_CPU_PROCESSES = 48

    # sets number of GPUs available
    N_GPUS = torch.cuda.device_count()

    # sets multiprocessing pool
    pool = Pool(N_CPU_PROCESSES)  

    # sets start method to spawn for GPU multiprocessing
    ctx = get_context('forkserver')

    # grabs all of the manifests
    manifests = glob.glob("../manifests/*.txt")

    # now we iterate over all of the manifests
    for manifest in manifests:
            
        # sets directory name
        dir_name = manifest.split('/')[-1][:-4] + "/"

        # first, we make the subdirectories for this manifest
        if not os.path.isdir(S3_SAVE_DIR + dir_name):
            os.mkdir(S3_SAVE_DIR + dir_name)
        if not os.path.isdir(OUTPUT_SAVE_DIR + dir_name):
            os.mkdir(OUTPUT_SAVE_DIR + dir_name)

        # read manifest
        page_filepaths = open(manifest, "r").read().split('\n')
        
        # remove duplicate filepaths
        page_filepaths = collections.Counter(page_filepaths).keys()
        
        # remove empty strings
        filtered_page_filepaths = []
        for filepath in page_filepaths:
            if filepath != '':
                filtered_page_filepaths.append(filepath)
        page_filepaths = filtered_page_filepaths
        
        # if there are no files in the manifest, we skip over this newspaper manifest
        if len(page_filepaths) == 0 or page_filepaths == ['']:
            print(manifest)

        print("PROCESSING MANIFEST: " + str(dir_name) + " (" + str(len(page_filepaths)) + " files)")

        print("validating filepaths...")

        # we check to ensure that all of these files exist; if some don't, we save the filepaths separately from the
        # main execution path
        packed_list = pool.map(files_exist, chunk(page_filepaths, N_CPU_PROCESSES))

        good_filepaths = []
        no_jp2 = []
        no_xml = []
        # we now unroll the lists from the different processes
        for contents in packed_list:
            good_filepaths.extend(contents[0])
            no_jp2.extend(contents[1])
            no_xml.extend(contents[2])

        # make sure all of the files have been tested
        assert len(page_filepaths) == len(good_filepaths) + len(no_jp2) + len(no_xml)

        # we now write this info to files for quick summarization
        with open(OUTPUT_SAVE_DIR + dir_name + 'processed_filepaths.txt', 'w') as f:
            for filepath in good_filepaths:
                f.write("%s\n" % filepath)

        with open(OUTPUT_SAVE_DIR + dir_name + 'no_jp2.txt', 'w') as f:
            for filepath in no_jp2:
                f.write("%s\n" % filepath)

        with open(OUTPUT_SAVE_DIR + dir_name + 'no_xml.txt', 'w') as f:
            for filepath in no_xml:
                f.write("%s\n" % filepath)

        # now we cd into the directory for the computations
        os.chdir(S3_SAVE_DIR + dir_name)
        
        # runs multiprocess for downloading of files in manifest
        print("retrieving files for manifest...")
        # chunks good filepaths for multiprocessing
        good_filepath_chunks = chunk(good_filepaths, N_CPU_PROCESSES)
        # adds directory to cd into (each process starts in local path of notebook)
        for i in range(0, len(good_filepath_chunks)):
            good_filepath_chunks[i] = [S3_SAVE_DIR + dir_name, good_filepath_chunks[i]]
        # calls the multiprocessing
        image_size_dicts = pool.map(retrieve_files, good_filepath_chunks)

        # we now combine the dictionaries into one
        image_size_dict = dict(ChainMap(*image_size_dicts))
        
        # now we generate predictions on all of the downloaded files
        print("predicting on pages...")
        
        # FOR MULTIPROCESSING
        chunked_image_filepaths = chunk(glob.glob("*.jpg"), N_GPUS)

        # https://stackoverflow.com/questions/31386613/python-multiprocessing-what-does-process-join-do
        processes = []
        for i in range(0, N_GPUS):
            zipped = [S3_SAVE_DIR, OUTPUT_SAVE_DIR, dir_name, INFERENCE_BATCH_SIZE, chunked_image_filepaths[i], i]
            p = ctx.Process(target=generate_predictions, args=(zipped,))
            p.start()
            processes.append(p)
            
        for process in processes:
            process.join()
            
        # now, we cd into the directory containing the output files
        os.chdir(OUTPUT_SAVE_DIR + dir_name)

        # now, we grab the JSON predictions and append on image width and height so the data can be zipped 
        # for multiprocessing
        # we want to pass these to the OCR retrieval function because they are necessary to compute bounding
        # boxes relative to METS/ALTO OCR, and opening the image using PIL or the equivalent is costly due to
        # the latency in loading the image into memory
        json_filepaths = glob.glob("*.json")

        # grabs the 
        json_info = []

        for json_filepath in json_filepaths:
            im_width, im_height = image_size_dict[json_filepath.replace('.json', '.jpg')]
            json_info.append([json_filepath, im_width, im_height]) 

        chunked_json_info = chunk(json_info, N_CPU_PROCESSES)
        for i in range(0, len(chunked_json_info)):
            chunked_json_info[i] = [OUTPUT_SAVE_DIR, dir_name, chunked_json_info[i]]

        print("grabbing OCR...")
        pool.map(retrieve_ocr, chunked_json_info) 
        
        print("cropping images...")
        zipped = chunk(json_filepaths, N_CPU_PROCESSES)
        for i in range(0, len(zipped)):
            zipped[i] = [OUTPUT_SAVE_DIR, S3_SAVE_DIR, dir_name, zipped[i]]
        pool.map(crop, zipped)

        print("generating embeddings...")
        
        # FOR MULTIPROCESSING
        chunked_json_filepaths = chunk(json_filepaths, N_GPUS)

        # https://stackoverflow.com/questions/31386613/python-multiprocessing-what-does-process-join-do
        processes = []
        for i in range(0, N_GPUS):
            zipped = [OUTPUT_SAVE_DIR, S3_SAVE_DIR, dir_name, chunked_json_filepaths[i], i]
            p = ctx.Process(target=generate_embeddings, args=(zipped,))
            p.start()
            processes.append(p)
            
        for process in processes:
            process.join()

        print("uploading...")
            
        # now, we cd back into the 'save' directory 
        os.chdir(OUTPUT_SAVE_DIR)

        # we now compute stats on the processed newspaper and save as json
        stats = {}
        paths = glob.glob(dir_name + "*.json")
        stats["processed_page_ct"] = len(paths)
        for path in paths:
            if "embeddings" in path:
                continue
            # loads the JSON
            with open(path) as f:
                data = json.load(f)
                stats[data["filepath"]] = data
           
        # save stats to file
        with open(dir_name + dir_name[:-1] + "_stats.json", "w") as fp:
            json.dump(stats, fp)
            
        # we write the JSON stats file to S3 bucket
        s3 = boto3.resource('s3')
        s3.Bucket('S3-BUCKET-HERE').upload_file(dir_name + dir_name[:-1] + "_stats.json", "chronam_stats/" + dir_name[:-1] + "_stats.json")
        os.remove(dir_name + dir_name[:-1] + "_stats.json")
        
        # now, we grab all of the files and upload to the S3 bucket in parallel
        all_paths = glob.glob("**/*", recursive=True)
        # we filter out directory paths and keep only filepaths
        filepaths = []
        for path in all_paths:
            if os.path.isfile(path):
                filepaths.append(path)
        # we now upload in parallel
        filepath_chunks = chunk(filepaths, N_CPU_PROCESSES)
        pool.map(upload_files, filepath_chunks)
        
        os.chdir(dir_name)
        
        # we now remove the folder & its contents to free up disk space
        for path in glob.glob("**/*.json", recursive=True):
            os.remove(path)
        for path in glob.glob("**/*.txt", recursive=True):
            os.remove(path)
        for path in glob.glob("**/*.jpg", recursive=True):
            os.remove(path)
        os.chdir('../')
        shutil.rmtree(os.getcwd() + "/" + dir_name)

        # navigate to the ChronAm pages, remove them (as well as the empty folder) 
        # and navigate back to the notebook directory
        os.chdir(S3_SAVE_DIR + dir_name)

        for path in glob.glob("*.xml"):
            os.remove(path)
        for path in glob.glob("*.jpg"):
            os.remove(path)

        os.chdir('../')
        os.rmdir(os.getcwd() + "/" + dir_name)
        os.chdir("../notebooks/")


