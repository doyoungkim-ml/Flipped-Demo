# /app.py
from flask import Flask, render_template, request
import requests
import torch 
from transformers import T5Tokenizer, T5ForConditionalGeneration

# model = T5ForConditionalGeneration.from_pretrained("seonghyeonye/flipped_3B", use_cuda=False, from_tf=False, force_download=True)
# tokenizer = T5Tokenizer.from_pretrained("seonghyeonye/flipped_3B", use_cuda=False, from_tf=False, force_download=True)
#Flask 객체 인스턴스 생성
app = Flask(__name__)

@app.route('/',methods=('GET', 'POST')) # 접속하는 url
def index():

    if request.method == 'POST':
        option = []
        i=1
        dictionary=request.form.to_dict()
        while f'options{i}' in dictionary.keys():
            label = dictionary[f'options{i}']
            if label != None:
                option.append(label)
                i+=1
            else:
                break
        info = {"instruction": request.form.get('instruction'), "input": request.form.get('input'), "option": option}
        print(info)
        res = requests.post('http://137.68.191.43:8080/receive', files=info).json() 
        return res
    return render_template('index.html', len = 0)

# @app.route('/classify',methods=('GET', 'POST'))
# def classify():
#     option = []
#     i=1

#     while True:
#         label = request.args.get(f'options{i}')
#         if label != None:
#             option.append(label)
#             i+=1
#         else:
#             break

    
#     info = {"instruction": request.args.get('instruction'), "input": request.args.get('input'), "option": option}
#     res = requests.post('http://137.68.191.43:8080/receive', files=info).json() 

#     return render_template('index.html',answer=answer, options=options, bar_color=bar_color, prob_list=prob_list, len = len(prob_list))
# def flipped(instruct, input, options):
#     print(instruct, input, options)
#     source_ids_batch=[]
#     target_ids_batch=[]
#     src_mask_batch=[]
#     target_mask_batch=[]
#     for target in options:
#         input_ = f'input: {input}\n output: {target}'
#         target_= instruct
#         print("input is:\n", input_)
#         print("target is:\n", target_)
#         source = tokenizer.batch_encode_plus([str(input_)], max_length=512, 
#                                                                 padding='max_length', truncation=True, return_tensors="pt")
#         targets = tokenizer.batch_encode_plus([str(target_)], max_length=64, 
#                                                         padding='max_length', truncation=True, return_tensors="pt")
#         new_options= options
#         source_ids = source["input_ids"].squeeze()
#         target_ids = targets["input_ids"].squeeze()
#         src_mask    = source["attention_mask"].squeeze()
#         target_mask = targets["attention_mask"].squeeze()
#         source_ids_batch.append(source_ids)
#         target_ids_batch.append(target_ids)
#         src_mask_batch.append(src_mask)
#         target_mask_batch.append(target_mask)
#     batch = {"source_ids":torch.stack(source_ids_batch).unsqueeze(dim=0), "source_mask": torch.stack(src_mask_batch).unsqueeze(dim=0), "target_ids": torch.stack(target_ids_batch).unsqueeze(dim=0), "target_mask": torch.stack(target_mask_batch).unsqueeze(dim=0)}
#     with torch.no_grad():
#         prob_list = []
#         for index in range(int(batch["source_ids"].shape[1])):
#             lm_channel_labels = batch["target_ids"][:,index,:]
#             lm_channel_labels[lm_channel_labels[:, :] ==tokenizer.pad_token_id] = -100
#             output_channel = model(
#                 input_ids=batch["source_ids"][:,index,:].contiguous(),
#                 attention_mask=batch["source_mask"][:,index,:].contiguous(),
#                 labels=lm_channel_labels.contiguous(),
#                 decoder_attention_mask=batch["target_mask"][:,index,:].contiguous()
#             )
#             logits_channel = batch["target_mask"][:,index,:].unsqueeze(-1) * torch.log_softmax(output_channel.logits, dim=-1)

#             lm_channel_labels=lm_channel_labels.unsqueeze(-1)
#             seq_token_log_prob=torch.zeros(lm_channel_labels.shape)
#             for i in range(lm_channel_labels.shape[0]):
#                 for j in range(lm_channel_labels.shape[1]):
#                     seq_token_log_prob[i][j][0] = logits_channel[i][j][lm_channel_labels[i][j][0]]
#             seq_channel_log_prob = seq_token_log_prob.squeeze(dim=-1).sum(dim=-1)
#             print(f'logit for label {options[index]} is {seq_channel_log_prob[0]}')
#             prob_list.append(seq_channel_log_prob.unsqueeze(-1))

#         concat = torch.cat(prob_list, 1).view(-1,int(batch["source_ids"].shape[1]))
#         probability = torch.exp(concat[0])
#         sum = torch.sum(probability)
#         probability /= sum
#         probability *=100
#         predictions = concat.argmax(dim=1)
#         return options[predictions.tolist()[0]], options, predictions.tolist()[0], probability.tolist()
if __name__=="__main__":
    app.run(host="0.0.0.0", port="8880",debug=True)
    # host 등을 직접 지정하고 싶다면
    # app.run(host="127.0.0.1", port="5000", debug=True)