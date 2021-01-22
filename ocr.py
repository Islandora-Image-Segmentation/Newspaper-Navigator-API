from lxml import etree as ET
from xml.etree.ElementTree import ElementTree
import json
import os

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


def retrieve_ocr(packet):
    # grab contents of packet, CD into correct directory
    dir_name = packet[1]
    os.chdir(packet[0] + dir_name)
    json_info = packet[2]
    S3_SAVE_DIR = packet[3]

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
