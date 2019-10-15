##!/usr/bin/env python3
import json
import os
import time
import traceback
import unicodedata

import fire
import numpy as np
import tensorflow as tf
from flask import Flask, request

import encoder
import model
import sample


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

    checkpoint = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, checkpoint)
    # this function ends after restoring the model - then only the inference part is served via REST


def single_step(raw_text, samples):
    print("completing '" + raw_text + "'...")
    start_time = time.time()
    initial_context = enc.encode(raw_text)
    output_contexts = sess.run(output, feed_dict={context: [initial_context for _ in range(samples)]})
    # Disabled, to keep the full paragraph and not just the added part
    # output_contexts = completed_context[:, len(initial_context):]
    output_texts = list(map(enc.decode, output_contexts))
    print("=" * 36 + " SAMPLE " + str(1) + " " + "=" * 36)
    print(output_texts)
    print("=" * 80 + ", Elapsed: " + str(time.time() - start_time))
    return output_texts, output_contexts


def run_app(http_port=1301):
    serve_model()
    single_step('This text is here to speed up the next inference. ', 1)
    app = Flask(__name__)

    @app.route('/v1/interactive', methods=['POST'])
    def single():
        try:
            initial_call_time = time.time()
            payload = request.get_json()
            in_text = payload['input']
            in_samples = payload['samples']
            if in_samples != 1:
                print('Resetting in_samples to 1, since the context was instantiated with batch 1')
                in_samples = 1
            output_texts, output_contexts = single_step(in_text, in_samples)
            for i in range(len(output_texts)):
                text = output_texts[i]
                text = text.split("<|endoftext|>")[0]
                text = text.strip()
                output_texts[i] = text
            response = {
                "input": in_text,
                "samples": in_samples,
                "completions": output_texts,
                "contexts": np.array_str(output_contexts),
                "backend_elapsed": time.time() - initial_call_time
            }
            return json.dumps(response), 200
        except Exception as e:
            print("EXCEPTION on /v1/interactive:")
            traceback.print_exc()
            response = {
                "backend_exception": repr(e)
            }
            return json.dumps(response), 500

    @app.route('/v1/control', methods=['POST'])
    def control():
        pass

    print("HTTP endpoint up and running. POST to /v1/interactive with {'input': 'your text..'}")
    # USE like:
    # curl -X POST -H "Content-Type: application/json" -d '{"input":"test"}' http://localhost:1301/v1/interactive
    app.run(host='127.0.0.1', port=http_port, threaded=False)


if __name__ == '__main__':
    fire.Fire(run_app)
