import time
from itertools import compress
from pytorch_lightning.utilities.cloud_io import load
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from utils.utils import parse_args, get_reader, load_model, get_out_filename, get_tagset
from utils.reader_utils import get_tags
from utils.emea import emea_ensemble

if __name__ == '__main__':
    timestamp = time.time()
    sg = parse_args()

    # load the dataset first
    test_data = get_reader(file_path=sg.test, target_vocab=get_tagset(sg.iob_tagging), max_instances=sg.max_instances, max_length=500, encoder_model=sg.encoder_model)
    
    # Add list of models.
    model_names = [] 
    
    models = []
    for file_name in model_names:
        models.append(load_model(file_name, tag_to_id=get_tagset(sg.iob_tagging)))

    model0 = load_model(model_names[0], tag_to_id=get_tagset(sg.iob_tagging))
    # use pytorch lightnings saver here.
    eval_file = "predictions/dev/naive/de_naive.conll"

    test_dataloaders = DataLoader(test_data, batch_size=25, collate_fn=model0.collate_batch, shuffle=False, drop_last=False)
    out_str = ''
    index = 0

    for batch in tqdm(test_dataloaders, total=len(test_dataloaders)):
        tokens, tags, mask, token_mask, metadata = batch
        scores = []
        for m in models:
            token_scores = m.perform_forward_step(batch, mode='ensemble_sm')
            scores.append(token_scores)
        
        if sg.emea=="true":
            ensemble_score = emea_ensemble(scores, steps=10, lr=500)
        else:
            ensemble_score = torch.sum(torch.stack(scores), dim=0)
        best_path = torch.argmax(ensemble_score, dim=2)

        pred_results, pred_tags = [], []
        for i in range(tokens.size(0)):
            tag_seq = best_path[i].tolist() # Softmax.
            pred_tags.append([model0.id_to_tag[x] for x in tag_seq])
            
        
        tag_results = [compress(pred_tags_, mask_) for pred_tags_, mask_ in zip(pred_tags, token_mask)]
        #return token_results, tag_results

        pred_tags = tag_results
        for pred_tag_inst in pred_tags:
            out_str += '\n'.join(pred_tag_inst)
            out_str += '\n\n'
        index += 1
    open(eval_file, 'wt').write(out_str)