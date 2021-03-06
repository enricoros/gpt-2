#!/usr/bin/env python3
import datetime
import json
import logging
import os
import sys
import time
import traceback
import unicodedata
from logging.handlers import RotatingFileHandler
from threading import Lock

import fire
import numpy as np
import tensorflow as tf
from flask import Flask, request
from flask.logging import default_handler as flask_default_logger
from werkzeug.exceptions import BadRequest

import encoder
import model
import sample

# Local constant to enable Execution tracing
PROFILE_ON = False
if PROFILE_ON is True:
    from tensorflow.python.client import timeline

# disable the TF warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# disable the TF deprecation messages
# tf._deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0
# enable XLA - doesn't give performance advantage
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

# was: length = None, top_k = 0,
def serve_model(model_name, seed=None, nsamples=1, batch_size=1, length=50, temperature=1, top_k=40,
                models_dir='models'):
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
    start_time = time.time()
    # Encode input
    initial_context = enc.encode(raw_text)
    encode_time = time.time()
    # Perform completions
    run_options = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE) if PROFILE_ON else None
    run_metadata = tf.compat.v1.RunMetadata() if PROFILE_ON else None
    try:
        output_contexts = sess.run(output,
                                   feed_dict={context: [initial_context for _ in range(samples)]},
                                   options=run_options, run_metadata=run_metadata)
    except:
        print("Flasky: Exception: likely resource exhaustion. Killing the process.")
        os.kill(os.getpid(), 9)
        # goodbye
    inference_time = time.time()
    # If profiling the execution, save the timeline to file in chrome-tracing format
    if run_metadata is not None:
        with open('timeline.json', 'w') as f:
            f.write(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())
    # Remove encoded input fom all outputs, since it's preserved in the response anyway
    output_contexts = output_contexts[:, len(initial_context):]
    # Decode the output context back to text
    output_texts = list(map(enc.decode, output_contexts))
    decode_time = time.time()
    return output_texts, output_contexts, inference_time - encode_time, encode_time - start_time


def set_gpu_number(phy_number):
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_visible_devices(gpu_list[phy_number], 'GPU')
        logical_gpu_list = tf.config.experimental.list_logical_devices('GPU')
        print("GPU selection:", len(gpu_list), "Physical GPUs,", len(logical_gpu_list), "Logical GPU")
    except IndexError as e:
        print("Incorrect 0-based Physical GPU index. There are " + str(len(gpu_list)) + " GPUs, starting from 0.")
        sys.exit()
    except RuntimeError as e:
        print("Visible devices must be set before GPUs have been initialized")
        sys.exit()


def set_gpu_memory(size_mb=None):
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpu_list:
            if size_mb is None:
                pass
            elif size_mb is 0:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("Set GPU " + gpu.name + " to Incremental memory allocation")
            else:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=size_mb)])
                print("Set GPU " + gpu.name + " to " + str(size_mb) + "MB max memory")
    except RuntimeError as e:
        print("Memory growth must be set before GPUs have been initialized")


def run_app(http_host='127.0.0.1', http_port=1301, model_name='774M', sample_size=1, length=50,
            gpu_off=None, gpu_phy=None, gpu_mem=None):
    # handle GPU selection and memory allocation
    if gpu_off is True:
        print("Disabling GPU support (forcing CPU)")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        if gpu_mem is not None:
            set_gpu_memory(gpu_mem)
        if gpu_phy is not None:
            set_gpu_number(gpu_phy)
    # restore the TensorFlow model
    serve_model(model_name, nsamples=sample_size, batch_size=sample_size, length=length)
    # run an inference to flush out kernels and speed up the real 1st inference
    single_step('This text is here to speed up the next inference. ', sample_size)
    # configure Flask for serving
    app = Flask(__name__)
    # configuring logging of various Flask requests
    formatter = logging.Formatter("[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")
    handler = RotatingFileHandler("flasky-serving-" + str(http_port) + ".log", maxBytes=10000000, backupCount=5)
    handler.setFormatter(formatter)
    app.logger.removeHandler(flask_default_logger)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.warning(' ========== SERVER STARTED ========== : ' + str(datetime.datetime.now()))
    single_tf_lock = Lock()

    @app.route('/v1/interactive', methods=['POST'])
    def single():
        try:
            # get and validate variable inputs
            initial_call_time = time.time()
            try:
                payload = request.get_json()
            except BadRequest:
                # happens if the JSON is malformed, for example
                raise Exception("body decode error")
            if payload is None:
                # happens if the Content-Type is not 'application/json'
                raise Exception("body format error")
            if 'input' not in payload:
                # happens if a critical attribute is missing
                raise Exception("body content error")
            in_text = payload['input']
            if not in_text:
                in_text = '<|endoftext|>\n'  # blank input translates to a context with only an end symbol
            in_samples = payload.get('samples', sample_size)
            if in_samples != sample_size:
                print('WARNING: Resetting in_samples to ' + str(sample_size) + ', to match session constraints')
                in_samples = sample_size

            # perform inference
            # app.logger.info('request: "' + in_text + '", length: ' + str(length) + ', samples: ' + str(in_samples))
            single_tf_lock.acquire()
            output_texts, output_contexts, inner_inference_time, encoding_time = single_step(in_text, in_samples)
            single_tf_lock.release()
            # app.logger.info('  in ' + str(inner_inference_time) + ' seconds')

            # cleanup completions
            output_texts = list(map(lambda x: unicodedata.normalize("NFKD", x.split("<|endoftext|>")[0]), output_texts))

            # log to console the request and response
            print()
            print("=" * 36 + " REQUEST " + "=" * 37)
            print(" >> (x" + str(in_samples) + ") '" + in_text + "'")
            for i in range(len(output_texts)):
                print("-" * 36 + " SAMPLE " + str(i) + " " + "-" * 36)
                try:
                    print(output_texts[i])  # this can cause an exception because the CP1252 format is assumed
                except Exception as fe:  # ignoring exceptions from format conversion
                    pass
            print("=" * 37 + " (" + str(round(inner_inference_time, 3)) + ") " + "=" * 36)

            # respond to the request
            return {
                       "input": in_text,
                       "samples": in_samples,
                       "completions": output_texts,
                       "backend_elapsed": time.time() - initial_call_time,
                       "backend_elapsed_inference": inner_inference_time,
                       # "backend_elapsed_encoding": encoding_time,
                   }, 200
        except Exception as e:
            print("EXCEPTION on /v1/interactive:")
            traceback.print_exc()
            return {"backend_exception": repr(e)}, 500

    @app.route('/v1/control', methods=['POST'])
    def control():
        return {"alive": True}, 200

    print("HTTP endpoint up and running. POST to /v1/interactive with {'input': 'your text..'}")
    # USE like:
    # curl -X POST -H "Content-Type: application/json" -d '{"input":"test "}' http://localhost:1301/v1/interactive
    app.run(host=http_host, port=http_port, threaded=True)


if __name__ == '__main__':
    fire.Fire(run_app)
