# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpretrain.registry import MODELS


def article(name):
    return 'an' if name[0] in 'aeiou' else 'a'


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace('_', ' ').replace('/', ' or ').lower()
    if rm_dot:
        res = res.rstrip('.')
    return res


single_template = ['a photo of a {}.']

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',
    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',
    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',
    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',
    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',
    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',
    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',
    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',
    'a plastic {}.',
    'the plastic {}.',
    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',
    'an embroidered {}.',
    'the embroidered {}.',
    'a painting of the {}.',
    'a painting of a {}.',
]

openimages_rare_unseen = [
    'Aerial photography', 'Aircraft engine', 'Ale', 'Aloe', 'Amphibian',
    'Angling', 'Anole', 'Antique car', 'Arcade game', 'Arthropod',
    'Assault rifle', 'Athletic shoe', 'Auto racing', 'Backlighting',
    'Bagpipes', 'Ball game', 'Barbecue chicken', 'Barechested', 'Barquentine',
    'Beef tenderloin', 'Billiard room', 'Billiards', 'Bird of prey',
    'Black swan', 'Black-and-white', 'Blond', 'Boating', 'Bonbon',
    'Bottled water', 'Bouldering', 'Bovine', 'Bratwurst', 'Breadboard',
    'Briefs', 'Brisket', 'Brochette', 'Calabaza', 'Camera operator', 'Canola',
    'Childbirth', 'Chordophone', 'Church bell', 'Classical sculpture',
    'Close-up', 'Cobblestone', 'Coca-cola', 'Combat sport', 'Comics',
    'Compact car', 'Computer speaker', 'Cookies and crackers',
    'Coral reef fish', 'Corn on the cob', 'Cosmetics', 'Crocodilia',
    'Digital camera', 'Dishware', 'Divemaster', 'Dobermann', 'Dog walking',
    'Domestic rabbit', 'Domestic short-haired cat', 'Double-decker bus',
    'Drums', 'Electric guitar', 'Electric piano', 'Electronic instrument',
    'Equestrianism', 'Equitation', 'Erinaceidae', 'Extreme sport', 'Falafel',
    'Figure skating', 'Filling station', 'Fire apparatus', 'Firearm',
    'Flatbread', 'Floristry', 'Forklift truck', 'Freight transport',
    'Fried food', 'Fried noodles', 'Frigate', 'Frozen yogurt', 'Frying',
    'Full moon', 'Galleon', 'Glacial landform', 'Gliding', 'Go-kart', 'Goats',
    'Grappling', 'Great white shark', 'Gumbo', 'Gun turret', 'Hair coloring',
    'Halter', 'Headphones', 'Heavy cruiser', 'Herding', 'High-speed rail',
    'Holding hands', 'Horse and buggy', 'Horse racing', 'Hound',
    'Hunting knife', 'Hurdling', 'Inflatable', 'Jackfruit', 'Jeans', 'Jiaozi',
    'Junk food', 'Khinkali', 'Kitesurfing', 'Lawn game', 'Leaf vegetable',
    'Lechon', 'Lifebuoy', 'Locust', 'Lumpia', 'Luxury vehicle', 'Machine tool',
    'Medical imaging', 'Melee weapon', 'Microcontroller', 'Middle ages',
    'Military person', 'Military vehicle', 'Milky way', 'Miniature Poodle',
    'Modern dance', 'Molluscs', 'Monoplane', 'Motorcycling', 'Musical theatre',
    'Narcissus', 'Nest box', 'Newsagent\'s shop', 'Nile crocodile',
    'Nordic skiing', 'Nuclear power plant', 'Orator', 'Outdoor shoe',
    'Parachuting', 'Pasta salad', 'Peafowl', 'Pelmeni', 'Perching bird',
    'Performance car', 'Personal water craft', 'Pit bull', 'Plant stem',
    'Pork chop', 'Portrait photography', 'Primate', 'Procyonidae',
    'Prosciutto', 'Public speaking', 'Racewalking', 'Ramen',
    'Rear-view mirror', 'Residential area', 'Ribs', 'Rice ball',
    'Road cycling', 'Roller skating', 'Roman temple', 'Rowing', 'Rural area',
    'Sailboat racing', 'Scaled reptile', 'Scuba diving', 'Senior citizen',
    'Shallot', 'Shinto shrine', 'Shooting range', 'Siberian husky', 'Sledding',
    'Soba', 'Solar energy', 'Sport climbing', 'Sport utility vehicle',
    'Steamed rice', 'Stemware', 'Sumo', 'Surfing Equipment', 'Team sport',
    'Touring car', 'Toy block', 'Trampolining', 'Underwater diving',
    'Vegetarian food', 'Wallaby', 'Water polo', 'Watercolor paint', 'Whiskers',
    'Wind wave', 'Woodwind instrument', 'Yakitori', 'Zeppelin'
]


def get_clip_model():
    model = dict(
        type='CLIPZeroShot',
        vision_backbone=dict(
            type='VisionTransformer',
            arch='base',
            img_size=224,
            patch_size=16,
            drop_rate=0.,
            layer_cfgs=dict(act_cfg=dict(type='mmpretrain.QuickGELU')),
            pre_norm=True,
        ),
        projection=dict(
            type='CLIPProjection', in_channels=768, out_channels=512),
        text_backbone=dict(
            type='CLIPTransformer',
            width=512,
            layers=12,
            heads=8,
            attn_mask=True,
        ),
        tokenizer=dict(
            type='AutoTokenizer',
            name_or_path='openai/clip-vit-base-patch16',
            use_fast=False),
        vocab_size=49408,
        transformer_width=512,
        proj_dim=512,
        context_length=77,
        data_preprocessor=dict(
            type='MultiModalDataPreprocessor',
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
            to_rgb=False,
        ),
    )
    return MODELS.build(model)


def build_openset_label_embedding(categories=None, clip_ckpt_path=''):
    if categories is None:
        print('Categories is None, so using rare_unseen categories')
        categories = openimages_rare_unseen
    model = get_clip_model()
    model.load_state_dict(torch.load(clip_ckpt_path))
    templates = multiple_templates

    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        openset_label_embedding = []
        for category in categories:
            texts = [
                template.format(
                    processed_name(category, rm_dot=True),
                    article=article(category)) for template in templates
            ]
            texts = [
                'This is ' + text
                if text.startswith('a') or text.startswith('the') else text
                for text in texts
            ]
            texts = model.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.extract_text_feat(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            openset_label_embedding.append(text_embedding)
        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)
        if run_on_gpu:
            openset_label_embedding = openset_label_embedding.cuda()

    openset_label_embedding = openset_label_embedding.t()
    return openset_label_embedding, categories
