import torch
import numpy as np
from visualize_attn import AttentionVisualizer
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
from captum.attr import visualization as viz

def tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length):
    tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True, max_length=max_input_length)
    labels = []
    label_mask = []

    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is not None:
            cur_label = original_labels[word_idx]
        if word_idx is None:
            labels.append(-100)
            label_mask.append(0)
        elif word_idx != previous_word_idx:
            labels.append(cur_label)
            label_mask.append(1)
        else:
            labels.append(-100)
            label_mask.append(0)
        previous_word_idx = word_idx

    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels, label_mask


def predict_r7(tokenizer, classification_model,  original_words, original_labels, max_input_length): #for one file
    n_correct = 0
    n_total = 0
    all_logits, predicted_labels = [], []
    for words, labels in zip(original_words, original_labels):
        input_ids, attention_mask, labels, labels_mask = tokenize_and_align_labels(tokenizer, words, labels, max_input_length)
        input_ids = torch.as_tensor(input_ids).unsqueeze(0).cuda()
        attention_mask = torch.as_tensor(attention_mask).unsqueeze(0).cuda()
        labels_tensor = torch.as_tensor(labels).unsqueeze(0).cuda()
        use_amp = False
        if use_amp:
            with torch.cuda.amp.autocast():
                _, logits = classification_model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
        else:
            _, logits = classification_model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
        preds = torch.argmax(logits, dim=2).squeeze().cpu().detach().numpy()
        numpy_logits = logits.squeeze(0).cpu().detach().numpy()
        all_logits.append(numpy_logits)
        for i, m in enumerate(labels_mask):
            if m:
                predicted_labels.append(preds[i])
                n_total += 1
                n_correct += preds[i] == labels[i]
    #print('Predictions: {} from Text: "{}"'.format(predicted_labels, original_words))
    print(n_correct, n_total)

def break_models(model):
    layers = list(model.children())
    middle_model = torch.nn.Sequential(*layers[:-2])
    rnn = False
    if middle_model[-1].__class__.__name__ in ['LSTM', 'GRU', 'RNN']:
        rnn = True
    #print(middle_model)
    return middle_model, rnn

def predict_r6(tokenizer, embedding, classification_model, texts, labels, max_input_length, cls_token_is_first=False, visualize=False, analyze=True):
    use_amp = False
    all_logits = []
    all_embeddings = []
    all_middle = []
    all_preds = []
    n_correct, n_sample = 0, 0
    #first_part, rnn = break_models(classification_model)
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
            if visualize:
                visualizer = AttentionVisualizer('trojai-r6', embedding, tokenizer)
                visualizer.display([text], input_ids=input_ids, attention_mask=attention_mask)
                #pass
            if analyze:
                pass
                #tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                #baseline = torch.ones_like(input_ids, dtype=torch.int64).cpu()*tokenizer.pad_token_id
                #gs = GradientShap(embedding.cpu())
                #attributions, delta = gs.attribute(torch.LongTensor(input_ids).clone().cpu(),  stdevs=0.09, n_samples=4, baselines=torch.LongTensor(baseline).cpu(), target=0, return_convergence_delta=True)
                #print(tokens)
                #print('GradientShap Attributions:', attributions)


            if cls_token_is_first: #TODO
                embedding_vector = embedding_vector[:, 0, :]
            else: #TODO for GPT remove padding
                embedding_vector = embedding_vector[:, -1, :]

            embedding_vector = embedding_vector.cpu().numpy()
            all_embeddings.append(embedding_vector)
            embedding_vector = np.expand_dims(embedding_vector, axis=0)
        #print(embedding_vector[:5])

        embedding_vector = torch.from_numpy(embedding_vector).cuda()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = classification_model(embedding_vector).cpu().detach().numpy()
                #if rnn:
                #    middle = first_part(embedding_vector)[0].squeeze(0).cpu().detach().numpy()
                #else:
                #    middle = first_part(embedding_vector).cpu().detach().numpy()

        else:
            logits = classification_model(embedding_vector).cpu().detach().numpy()
            #if rnn:
            #    middle = first_part(embedding_vector)[0].squeeze(0).cpu().detach().numpy()
            #else:
            #    middle = first_part(embedding_vector).cpu().detach().numpy()

        all_logits.append(logits)
        #all_middle.append(middle)
        sentiment_pred = np.argmax(logits)
        all_preds.append(sentiment_pred)
        n_sample += 1
        if sentiment_pred == label:
            n_correct += 1
        else:
            #print(idx, sentiment_pred, label)
            pass
    print(n_correct, n_sample, n_correct/n_sample)
    all_logits = np.concatenate(all_logits, axis=0)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_preds = np.array(all_preds)
    #all_middle = np.concatenate(all_middle, axis=0)
    return all_logits, all_preds



