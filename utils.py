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

def crop(image: Image.Image, box):
  # Use built in crop function for PIL Image. Box co-ordinates converted to image co-ordinates
  cropped = image.crop((box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height)).convert('RGB')

  return cropped


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