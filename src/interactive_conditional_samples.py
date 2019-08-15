##!/usr/bin/env python3
import fire
import json
import os
import numpy as np
import tensorflow as tf
import time

import model, sample, encoder

# disable the TF warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# disable the TF deprecation messages
tf._deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0
# enable XLA - doesn't give performance advantage
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'


SOUTH_LINES = {}
def parse_sp_episodes(fileName):
    with open(fileName, newline='') as csvfile:
        import csv
        for row in csv.reader(csvfile):
            if len(row) == 1 or row[0] == 'Season':
                continue
            assert len(row) == 4
            season = int(row[0])
            episode = int(row[1])
            character = row[2]
            line = row[3]
            SOUTH_LINES.setdefault(season, {}).setdefault(episode, []).append([character, line])

def make_sp_text(lines, season, episode, skip, count, extra):
    transcripts = lines[season][episode]
    text = ""
    for line in transcripts:
        skip = skip - 1
        if skip >= 0:
            continue
        text += line[0] + ": " + line[1] + extra
        count = count - 1
        if count < 1:
            break
    return text


def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """

    # Robert Miles - YouTube comments config
    nsamples = 1
    length = 150
    top_k = 40

    # Enrico's
    # nsamples = 4
    # batch_size = 4
    nsamples = 2
    batch_size = 2
    length = 100
    top_k = 40


    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            # if the text entered is "sp", feed a primer conversation from South Park
            if raw_text == "sp":
                txt_season = 18
                txt_episode = 1
                print(" Using South Park season " + str(txt_season) + " ep " + str(txt_episode) + " !!")
                if txt_season not in SOUTH_LINES:
                    parse_sp_episodes('Season-' + str(txt_season) + '.csv')
                raw_text = make_sp_text(SOUTH_LINES, txt_season, txt_episode, 135, 30, "\n")
                raw_text += "Cartman: "
                print(raw_text)
            print("got it.. processing")
            start_time = time.time()
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                print("done. Samples below:")
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80 + ", Elapsed: " + str(time.time() - start_time))

if __name__ == '__main__':
    fire.Fire(interact_model)

