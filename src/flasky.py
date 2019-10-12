##!/usr/bin/env python3
import fire
import json
import os
import numpy as np
import tensorflow as tf
import time

import encoder, model, sample
from flask import Flask, request

# disable the TF warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# disable the TF deprecation messages
# tf._deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0
# enable XLA - doesn't give performance advantage
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'


def serve_model(model_name='774M', seed=None, nsamples=1, batch_size=1, length=50, temperature=1, top_k=40,
                models_dir='models', raw_text=None, ):
    global enc
    global sess
    global output
    global context
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

    sess = tf.compat.v1.Session(graph=tf.Graph())
    sess.__enter__()
    context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k
    )

    ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, ckpt)
    # this function ends after restoring the model - then only the inference part is served via REST


def single_step(raw_text, samples):
    print("completing '" + raw_text + "'...")
    start_time = time.time()
    input_tokens = enc.encode(raw_text)
    out_all = sess.run(output, feed_dict={context: [input_tokens]})
    print(out_all)
    out = out_all[:, len(input_tokens):]
    print(out)
    text = enc.decode(out[0])
    print("=" * 36 + " SAMPLE " + str(1) + " " + "=" * 36)
    print(text)
    print("=" * 80 + ", Elapsed: " + str(time.time() - start_time))
    return text


def run_app():
    serve_model()
    single_step('Warm me up', 1)
    app = Flask(__name__)

    @app.route('/v1/interactive', methods=['POST'])
    def classify():
        try:
            payload = request.get_json()
            in_text = payload['input']
            in_samples = payload['samples']
            out_text = single_step(in_text, in_samples)
            return out_text, 200
        except Exception as e:
            return repr(e), 500

    app.run(host='127.0.0.1', port=1301)


if __name__ == '__main__':
    fire.Fire(run_app)
