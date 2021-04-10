import tensorflow as tf
import numpy as np
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig, PegasusForConditionalGeneration, PegasusTokenizer
from model import create_model
from preprocess import BertData, Processor
from data_loader import transfer_examples_to_inputs
import torch

model_save_path= os.path.join(os.getcwd(), "saved_model")


# api for market sentiment analysis #

def get_mkst(text, model_save_path):

    model_save_path = model_save_path

    mkst_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    mkst_model = TFBertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

    mkst_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mkst_metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    mkst_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

    mkst_model.compile(loss=mkst_loss,optimizer=mkst_optimizer, metrics=[mkst_metric])
    mkst_model_path = os.path.join(model_save_path, "mkst", "mkst.h5")

    mkst_model.load_weights(mkst_model_path)


    text_tokens = mkst_tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, pad_to_max_length = True, return_attention_mask=True)
    input_ids = text_tokens["input_ids"]
    attention_mask = text_tokens["attention_mask"]
    input = [input_ids, attention_mask]
    result = mkst_model.predict(input)
    res = np.argmax(result[0])

    return res


# api for extrative summarization(keras english)

def get_extsum_eng(text, model_save_path, Processor, rank=3):

    ans = []

    def get_predict_input(input_data):

        input_ids = input_data[0]
        labels = input_data[-1]
        segment_ids = input_data[1]
        cls_ids = input_data[3]
        mask_cls = input_data[4]
        mask = input_data[2]
        helper = np.arange(0, 1, dtype=int)[:, None]

        x = [input_ids, segment_ids, cls_ids, mask, mask_cls, helper, labels]

        return x

    ckpt_path = os.path.join(model_save_path, "bertsumext")

    # load model
    bert_sum = create_model("bert-base-uncased", 512, "transformer", 768, 512, 8, 0.1, 2)
    bert_sum.load_weights(ckpt_path)

    # # show the model
    bert_sum.summary()

    # # check the accuracy and loss
    # loss, acc = bert_sum.evaluate()

    # input_text, bert_name, min_src_ntokens, max_src_ntokens, max_nsents, min_nsents, max_length
    # Processor = Processor("bert-base-uncased", 3, 200, 100, 3, 512, "predict")
    example, src = Processor.preprocess(text, "", oracle_ids=[])
    input = transfer_examples_to_inputs([example], is_test=False)
    #
    x = get_predict_input(input)
    # print(x)
    result = bert_sum.predict(x)
    # print("Result:", result)

    ind = np.argsort(-result).astype(np.int32)
    # print("ind", ind)

    for i in range(rank):
        print("Summ",i,": ", src[ind[0][i]])
        ans.append(src[ind[0][i]])

    return "".join(ans)


# api for extractive summarization(japanese)

# PAGERANK ALGORITHMS


# api for abstractive summarization(pegasus, english)

def pegasus(text):

    src_text = [text]
    model_name = 'google/pegasus-xsum'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(torch_device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return tgt_text[0]

# api for NER
