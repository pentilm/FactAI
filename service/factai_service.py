# Copyright 2017 Benjamin Riedel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Import relevant packages and modules
from service.util import *
import random
import tensorflow as tf

from http.server import BaseHTTPRequestHandler, HTTPServer
import json

import sys
import grpc
import time
from concurrent import futures
#sys.path.append("./service_spec")
import service.service_spec.factai_service_pb2 as pb2
import service.service_spec.factai_service_pb2_grpc as pb2_grpc

mode = None
serve_mode = None
serve_port = None

if len(sys.argv) == 4:
    mode = sys.argv[1]
    serve_mode = sys.argv[2]
    serve_port = int(sys.argv[3])
else:
    # Prompt for mode
    mode = input('mode (serve / load / train)? ')

# Set file names
file_train_instances = "service/train_stances.csv"
file_train_bodies = "service/train_bodies.csv"
file_test_instances = "service/test_stances_unlabeled.csv"
file_test_bodies = "service/test_bodies.csv"
file_predictions = 'service/predictions_test.csv'


# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90


# Load data sets
raw_train = FNCData(file_train_instances, file_train_bodies)
raw_test = FNCData(file_test_instances, file_test_bodies)
n_train = len(raw_train.instances)


# Process data sets
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
feature_size = len(train_set[0])
print("feature_size: ", feature_size)
test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)


# Define model

# Create placeholders
features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
stances_pl = tf.placeholder(tf.int64, [None], 'stances')
keep_prob_pl = tf.placeholder(tf.float32)

# Infer batch size
batch_size = tf.shape(features_pl)[0]

# Define multi-layer perceptron
hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
logits = tf.reshape(logits_flat, [batch_size, target_size])

# Define L2 loss
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

# Define overall loss
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, stances_pl) + l2_loss)

# Define prediction
softmaxed_logits = tf.nn.softmax(logits)
predict = tf.arg_max(softmaxed_logits, 1)


class GRPCapi(pb2_grpc.FACTAIStanceClassificationServicer):
    def __init__(self, tf_session):
        self.tf_session = tf_session

    def stance_classify(self, req, ctxt):
        headline = req.headline
        body = req.body
        input_data = {'headline' : headline,
                      'body' : body}
        test_set = pipeline_serve(input_data,
                                  bow_vectorizer,
                                  tfreq_vectorizer,
                                  tfidf_vectorizer)
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        pred = self.tf_session.run(softmaxed_logits, feed_dict=test_feed_dict)[0]
        stance_pred = pb2.Stance()
        stance_pred.agree = pred[0]
        stance_pred.disagree = pred[1]
        stance_pred.discuss = pred[2]
        stance_pred.unrelated = pred[3]
        return stance_pred

def run_server(tf_session):
    class HTTPapi(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            content_type = self.headers['Content-Type']
            if content_type == 'application/json':
                data = self.rfile.read(content_length)
                json_data = json.loads(data.decode('utf-8'))
                if list(json_data.keys()).count('headline') == 1 and \
                   list(json_data.keys()).count('body') == 1:
                    input_data = {'headline' : json_data['headline'],
                                  'body' : json_data['body']}
                    test_set = pipeline_serve(input_data,
                                              bow_vectorizer,
                                              tfreq_vectorizer,
                                              tfidf_vectorizer)
                    test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
                    pred = tf_session.run(softmaxed_logits, feed_dict=test_feed_dict)[0]
                    labeled_pred = zip(["agree", "disagree", "discuss", "unrelated"],
                                       ['{:.2f}'.format(s_c) for s_c in pred])
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(json.dumps(dict(labeled_pred)).encode('utf-8'))
                else:
                    self.send_response(400)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write("Input should be json with keys 'headline' and 'body'")
            else:
                self.send_response(400)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write("Input should be json with keys 'headline' and 'body'")
    return HTTPapi


if mode == 'serve':
    sess = tf.Session()
    load_model(sess)
    if serve_port == None:
        serve_port = 7007
    if serve_mode == None:
        serve_mode = input('input (rest / grpc)? ')
    if serve_mode == 'rest':
        serve_address = ''
        server_handler = run_server(sess)
        httpd = HTTPServer((serve_address, serve_port), server_handler)
        try:
            print("Starting Server on port: " + str(serve_port))
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Exiting....")
            httpd.server_close()
    elif serve_mode == 'grpc':
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        pb2_grpc.add_FACTAIStanceClassificationServicer_to_server(GRPCapi(sess), grpc_server)
        grpc_server.add_insecure_port('[::]:' + str(serve_port))
        grpc_server.start()
        print("GRPC Server Started on port: " + str(serve_port))
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("Exiting....")
            grpc_server.stop(0)


# Load model
if mode == 'load':
    with tf.Session() as sess:
        load_model(sess)

        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)
        
        # Save predictions
        save_predictions(test_pred, file_predictions)


# Train model
if mode == 'train':

    # Define optimiser
    opt_func = tf.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

    # Perform training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            total_loss = 0
            indices = list(range(n_train))
            r.shuffle(indices)

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_features = [train_set[i] for i in batch_indices]
                batch_stances = [train_stances[i] for i in batch_indices]

                batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
                _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                total_loss += current_loss


        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)

        # Save predictions
        save_predictions(test_pred, file_predictions)

