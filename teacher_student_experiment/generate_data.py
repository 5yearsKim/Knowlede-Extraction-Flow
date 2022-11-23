import torch
from data_handler import PseudoGenerator, NoisePseudoGenerator, KegnetPseudoGenerator
from KEflow.model import prepare_classifier, prepare_flow
from kegnet.generator.utils import init_generator
from config import TYPE_DATA, DISTRIBUTION, im_size, cls_type, NUM_BATCH, BATCH_SIZE


classifier = prepare_classifier(cls_type, im_size[0], 10)
cls_state_dict = torch.load(f'ckpts/teacher/classifier_{TYPE_DATA.lower()}_{cls_type.lower()}.pt')
classifier.load_state_dict(cls_state_dict["model_state"])

if DISTRIBUTION in ["UNIFORM" , "NORMAL"]:
    data_generator = NoisePseudoGenerator(classifier, im_size, 10, temp=1., log_path=f'pseudo_data/{DISTRIBUTION.lower()}_{TYPE_DATA.lower()}.json', prior_type=DISTRIBUTION.lower())

if DISTRIBUTION in ["KEFLOW", "KEFLOW+"]:
    # define models / load classifier
    flow = prepare_flow('NICE', im_size[0], 10)

    # load checkpoint
    if DISTRIBUTION == 'KEFLOW+':
        name_flow = 'aided_keflow'
    if DISTRIBUTION == 'KEFLOW':
        name_flow = 'keflow'

    flow_state_dict = torch.load(f"ckpts/generator/{name_flow}_{TYPE_DATA.lower()}.pt")
    flow.load_state_dict(flow_state_dict["model_state"])

    # Data_generation
    data_generator = PseudoGenerator(flow, classifier, im_size, 10, temp=0.5, log_path=f"./pseudo_data/{DISTRIBUTION.lower()}_{TYPE_DATA.lower()}.json")

if DISTRIBUTION == 'KEGNET':
    if TYPE_DATA == 'DIGIT':
        dataset = 'mnist'
    if TYPE_DATA in ['FASHION', 'SVHN']:
        dataset = TYPE_DATA.lower()
    # define model
    kegnet = init_generator(dataset)

    kegnet_state_dict = torch.load(f'ckpts/generator/kegnet_{TYPE_DATA.lower()}.pt')
    kegnet.load_state_dict(kegnet_state_dict["model_state"])

    data_generator = KegnetPseudoGenerator(kegnet, classifier, log_path=f'./pseudo_data/kegnet_{TYPE_DATA.lower()}.json')

print(f'{TYPE_DATA} {DISTRIBUTION}')
data_generator.generate(NUM_BATCH, BATCH_SIZE)


