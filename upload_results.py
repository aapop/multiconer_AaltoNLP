import pandas as pd

from utils.toolkits import gsio
from utils.toolkits import load_config_yml

if __name__ == "__main__":
    # Create a column
    df = pd.read_csv("", sep='\t', header=None, )
    df.columns = ['metric', 'value']
    df = df.sort_values(by=['metric'], ascending=False).copy()

    data = (df.set_index('metric').to_dict('index'))
    data = list(data.values())
    metrics = [k['value'] for k in data]
    exp_name = ""
    text_encoder = "xlm-roberta-large"
    decoder = "softmax"
    lr_encoder = 1e-6
    lr_decoder = 1e-3
    dropout = 0.1
    total_steps = ""
    warmup_steps = ""
    post_hoc = "no"
    datalist = [exp_name, text_encoder, decoder, lr_encoder, lr_decoder,
        dropout, total_steps, warmup_steps, post_hoc] + metrics


    credential = '../credentials/multiconer.json'
    sheet_name = "multiconer"
    gs = gsio(credential, sheet_name)
    gs.read_sheet('')
    gs.insert_row(sheet_title='', row_data=datalist)