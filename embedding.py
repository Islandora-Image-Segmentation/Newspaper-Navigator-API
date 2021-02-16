# %% [markdown]
# # This cell defines a function for generating embeddings of each predicted box:

# %%
from img2vec_pytorch import Img2Vec
from PIL import Image
import torch
import json

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
            jpg_filepath = S3_SAVE_DIR + dir_name + \
                json_filepath.replace('.json', '.jpg')

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
                embedding_resnet_50 = img2vec_resnet_50.get_vec(
                    im, tensor=False)
                embedding_resnet_18 = img2vec_resnet_18.get_vec(
                    im, tensor=False)
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
