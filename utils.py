import os
import json
import math
import io
import subprocess

import s3fs
import boto3
import botocore
from PIL import Image

from config import OUTPUT_SAVE_DIR

def crop(packet):
  OUTPUT_SAVE_DIR = packet[0]
  S3_SAVE_DIR = packet[1]
  dir_name = packet[2]
  json_filepaths = packet[3]

  os.chdir(OUTPUT_SAVE_DIR + dir_name)

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
      cropped = im.crop((box[0] * im.width, box[1] * im.height, box[2] * im.width, box[3] * im.height)).convert('RGB')
      # save cropped image to output directory
      cropped_filepath = json_filepath.replace(".json", "_" + str(i).zfill(3) + "_" + str(pred_class) + "_" + str(
        int(math.floor(100 * score))).zfill(2) + ".jpg")
      cropped.save(cropped_filepath)
      new_filepath = dir_name + "data/" + cropped_filepath.split("data_")[1].replace(dir_name, '').replace('_', '/')
      new_filepath = new_filepath[:new_filepath.rfind("/")] + "_" + new_filepath[new_filepath.rfind("/") + 1:]
      new_filepath = new_filepath[:new_filepath.rfind("/")] + "_" + new_filepath[new_filepath.rfind("/") + 1:]
      content_filepaths.append(new_filepath)

    # add filepaths of extracted visual content to output
    predictions['visual_content_filepaths'] = content_filepaths

    # we save the updated JSON
    with open(json_filepath, 'w') as f:
      json.dump(predictions, f)


def chunk(file_list, n_chunks):
  # make chunks of files to be distributed across processes
  chunks = []
  chunk_size = math.ceil(float(len(file_list)) / n_chunks)
  for i in range(0, n_chunks - 1):
    chunks.append(file_list[i * chunk_size:(i + 1) * chunk_size])
  chunks.append(file_list[(n_chunks - 1) * chunk_size:])

  return chunks


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
          im.resize((math.floor(im.width / 6), math.floor(im.height / 6)), resample=0).save(
            local_filepath.replace(".jp2", ".jpg"))
          im_size_dict[local_filepath.replace(".jp2", ".jpg")] = (im.width, im.height)

        # if the JPEG doesn't exist, pull down the JPEG-2000 from 'ndnp-batches' S3 bucket and convert to JPG
        else:
          s3.Bucket('ndnp-batches').download_file(page_filepath, local_filepath)
          subprocess.call('gm convert ' + local_filepath + ' ' + local_filepath.replace(".jp2", ".jpg"), shell=True)
          subprocess.call('rm ' + local_filepath, shell=True)

          if not os.path.exists(local_filepath.replace(".jp2", ".jpg")):
            continue

          im = Image.open(local_filepath.replace(".jp2", ".jpg"))
          im.resize((math.floor(im.width / 6), math.floor(im.height / 6)), resample=0).save(
            local_filepath.replace(".jp2", ".jpg"))
          im_size_dict[local_filepath.replace(".jp2", ".jpg")] = (im.width, im.height)

      except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
          print("The object does not exist.")
        else:
          print("Error in downloading JP2.")
          raise

      try:
        s3.Bucket('ndnp-batches').download_file(page_filepath.replace(".jp2", ".xml"),
                                                local_filepath.replace(".jp2", ".xml"))
      except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
          print("The object does not exist.")
        else:
          print("Error in downloading XML.")
          raise

    return im_size_dict


def upload_files(filepaths):
  # cd into the 'save' directory
  os.chdir(OUTPUT_SAVE_DIR)

  # connects to boto3
  s3 = boto3.resource('s3')

  # uploads files to mirror the ChronAm structure
  for filepath in filepaths:

    if '_stats.json' in filepath or '.txt' in filepath:
      s3.Bucket("S3-BUCKET-HERE").upload_file(filepath, "chronam_processed/" + filepath)
      continue

    if '.json' in filepath:
      dir_name = filepath.split("/")[0] + "/"
      new_filepath = dir_name + "data/" + filepath.split("data_")[1].replace(dir_name, '').replace('_', '/')
      s3.Bucket("S3-BUCKET-HERE").upload_file(filepath, "chronam_processed/" + new_filepath)
      continue

    if '.jpg' in filepath:
      dir_name = filepath.split("/")[0] + "/"
      new_filepath = dir_name + "data/" + filepath.split("data_")[1].replace(dir_name, '').replace('_', '/')
      new_filepath = new_filepath[:new_filepath.rfind("/")] + "_" + new_filepath[new_filepath.rfind("/") + 1:]
      new_filepath = new_filepath[:new_filepath.rfind("/")] + "_" + new_filepath[new_filepath.rfind("/") + 1:]
      s3.Bucket("S3-BUCKET-HERE").upload_file(filepath, "chronam_processed/" + new_filepath)
      continue