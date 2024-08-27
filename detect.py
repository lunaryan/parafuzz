#Note: This file mainly contains the forward pass when feeding sentences into the model. The example code is implemented 
#for TrojAI competition round 6 dataset. You can change it to be suitable for your own dataset. The function should return 
#logits (ndarray) and predictions (ndarray) of the input sentences.

import torch
import numpy as np

def predict_r6(tokenizer, embedding, classification_model, texts, labels, max_input_length, cls_token_is_first=False):
    use_amp = False
    all_logits = []
    all_embeddings = []
    all_middle = []
    all_preds = []
    n_correct, n_sample = 0, 0
    for idx, (text, label)  in enumerate(zip(texts, labels)):
        results = tokenizer(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
        # extract the input token ids and the attention mask
        input_ids = results.data['input_ids'].cuda()
        attention_mask = results.data['attention_mask'].cuda()

        # convert to embedding
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    embedding_vector = embedding(input_ids, attention_mask=attention_mask, output_attentions=True)[0]
            else:
                embedding_vector = embedding(input_ids, attention_mask=attention_mask, output_attentions=True)[0]

            if cls_token_is_first: 
                embedding_vector = embedding_vector[:, 0, :]
            else: # for GPT remove padding
                embedding_vector = embedding_vector[:, -1, :]

            embedding_vector = embedding_vector.cpu().numpy()
            all_embeddings.append(embedding_vector)
            embedding_vector = np.expand_dims(embedding_vector, axis=0)

        embedding_vector = torch.from_numpy(embedding_vector).cuda()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = classification_model(embedding_vector).cpu().detach().numpy()
        else:
            logits = classification_model(embedding_vector).cpu().detach().numpy()

        all_logits.append(logits)
        sentiment_pred = np.argmax(logits)
        all_preds.append(sentiment_pred)
        n_sample += 1
        if sentiment_pred == label:
            n_correct += 1
        else:
            pass
    print(n_correct, n_sample, n_correct/n_sample)
    all_logits = np.concatenate(all_logits, axis=0)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_preds = np.array(all_preds)

    return all_logits, all_preds



