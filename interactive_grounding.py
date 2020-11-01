from types import SimpleNamespace
from typing import Tuple

import PIL
import torch

import numpy as np
import matplotlib.pyplot as plt
from pytorch_transformers import BertTokenizer

from script.extract_features import FeatureExtractor as BaseFeatureExtractor
from vilbert.vilbert import BertConfig, VILBertForVLTasks

BERT_CONFIG = 'config/bert_base_6layer_6conect.json'
PRETRAINED_BERT = 'bert-base-uncased'


class FeatureExtractor(BaseFeatureExtractor):

    def __init__(self):
        self.args = self.get_parser()
        self.detection_model = self._build_detection_model()

    def get_parser(self):
        return SimpleNamespace(model_file='data/detectron_model.pth',
                               config_file='data/detectron_config.yaml',
                               batch_size=1,
                               num_features=100,
                               feature_name="fc6",
                               confidence_threshold=0,
                               background=False,
                               partition=0)


def prediction(img, model, question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, ):
    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, attn_data_list = model(
        question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens,
        output_all_attention_masks=True
    )

    height, width = img.shape[0], img.shape[1]

    logits = torch.max(vil_prediction, 1)[1].data  # argmax

    # grounding:
    logits_vision = torch.max(vision_logit, 1)[1].data
    grounding_val, grounding_idx = torch.sort(vision_logit.view(-1), 0, True)

    examples_per_row = 5
    ncols = examples_per_row
    nrows = 1
    figsize = [12, ncols]  # figure size, inches
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, axi in enumerate(ax.flat):
        idx = grounding_idx[i]
        val = grounding_val[i]
        box = spatials[0][idx][:4].tolist()
        y1 = int(box[1] * height)
        y2 = int(box[3] * height)
        x1 = int(box[0] * width)
        x2 = int(box[2] * width)
        patch = img[y1:y2, x1:x2]
        axi.imshow(patch)
        axi.axis('off')
        axi.set_title(str(i) + ": " + str(val.item()))

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def custom_prediction(img, model, query, task, features, infos, tokenizer):

    tokens = tokenizer.encode(query)
    tokens = tokenizer.add_special_tokens_single_sentence(tokens)

    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(tokens)

    max_length = 37
    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [0] * (max_length - len(tokens))
        tokens = tokens + padding
        input_mask += padding
        segment_ids += padding

    text = torch.from_numpy(np.array(tokens)).cuda().unsqueeze(0)
    input_mask = torch.from_numpy(np.array(input_mask)).cuda().unsqueeze(0)
    segment_ids = torch.from_numpy(np.array(segment_ids)).cuda().unsqueeze(0)
    task = torch.from_numpy(np.array(task)).cuda().unsqueeze(0)

    num_image = len(infos)

    feature_list = []
    image_location_list = []
    image_mask_list = []
    for i in range(num_image):
        image_w = infos[i]['image_width']
        image_h = infos[i]['image_height']
        feature = features[i]
        num_boxes = feature.shape[0]

        g_feat = torch.sum(feature, dim=0) / num_boxes
        num_boxes = num_boxes + 1
        feature = torch.cat([g_feat.view(1,-1), feature], dim=0)
        boxes = infos[i]['bbox']
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:,:4] = boxes
        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)
        g_location = np.array([0,0,1,1,1])
        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
        image_mask = [1] * (int(num_boxes))

        feature_list.append(feature)
        image_location_list.append(torch.tensor(image_location))
        image_mask_list.append(torch.tensor(image_mask))

    features = torch.stack(feature_list, dim=0).float().cuda()
    spatials = torch.stack(image_location_list, dim=0).float().cuda()
    image_mask = torch.stack(image_mask_list, dim=0).byte().cuda()
    co_attention_mask = torch.zeros((num_image, num_boxes, max_length)).cuda()

    prediction(img, model, text, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task)



def main():
    config = BertConfig.from_json_file(BERT_CONFIG)
    device = torch.device("cuda")

    model = VILBertForVLTasks.from_pretrained(
        PRETRAINED_BERT,
        config=config,
        num_labels=1,
        default_gpu=True,
    )

    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(
        PRETRAINED_BERT, do_lower_case=True,
    )

    while True:
        try:
            image_path = input('Image:\n').strip()
            query = input('Query:\n').strip()
            features, info = image_features(image_path)
            print(features, info)

            img = PIL.Image.open(image_path).convert('RGB')
            img = torch.tensor(np.array(img))

            task = [9]

            custom_prediction(img, model, query, task, features, info, tokenizer=tokenizer)
        except Exception as e:
            print(e)


def image_features(image_path: str) -> Tuple[list, list]:
    feature_extractor = FeatureExtractor()

    features, info = feature_extractor.get_detectron_features([image_path])
    return features, info


if __name__ == '__main__':
    main()
