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
import os

from http.server import BaseHTTPRequestHandler, HTTPServer
import json

import sys
import grpc
import time
from concurrent import futures
#sys.path.append("./service_spec")
import service.service_spec.factai_service_pb2 as pb2
import service.service_spec.factai_service_pb2_grpc as pb2_grpc
from resutils import *

import logging
import log

from run_factai_service import Log

import service.service_spec.service_proto_pb2 as service_proto_pb2
import service.service_spec.service_proto_pb2_grpc  as service_proto_pb2_grpc

import threading


from time import sleep


import opentelemetry
from opentelemetry import trace
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_span_in_context


jaeger_address = os.environ['jaeger_address']


jaeger_exporter = JaegerExporter(
    agent_host_name=jaeger_address,
    agent_port=6831,
)


trace_provider=trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

grpc_server_instrumentor = GrpcInstrumentorServer()
grpc_server_instrumentor.instrument()

tracer_server=opentelemetry.instrumentation.grpc.server_interceptor(tracer_provider=trace_provider)
tracer=trace.get_tracer(__name__)



try:
    grpc_port = os.getenv("NOMAD_PORT_rpc")
except:
    grpc_port = "7007"    

try:
    tokenomics_mode=os.environ['tokenomics_mode']
except:
    tokenomics_mode=""

logger=Log.logger

serve_port = os.environ['SERVICE_PORT']

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
logger.info("feature_size: "+ str(feature_size))
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
        current_span = trace.get_current_span()
        current_span.set_attribute("http.route", "some_route")
        sleep(30 / 1000)
        try:
            telemetry=resutils()
            start_time=time.time()
            cpu_start_time=telemetry.cpu_ticks()
        except Exception as e:
            logger.error(e)
        
        headline = req.headline
        body = req.body
        call_id = req.call_id
        input_data = {'headline' : headline,
                      'body' : body}
        test_set = pipeline_serve(input_data,
                                  bow_vectorizer,
                                  tfreq_vectorizer,
                                  tfidf_vectorizer)
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        pred = self.tf_session.run(softmaxed_logits, feed_dict=test_feed_dict)[0]
        stance_pred = pb2.Stance()
        resp=pb2.Resp()
        stance_pred.agree = pred[0]
        stance_pred.disagree = pred[1]
        stance_pred.discuss = pred[2]
        stance_pred.unrelated = pred[3]
        response=""
        current_span.add_event("event message", {"prediction": str(pred)})
        try:
            memory_used=telemetry.memory_usage()
            time_taken=time.time()-start_time
            cpu_used=telemetry.cpu_ticks()-cpu_start_time
            net_used=telemetry.block_in()
            result = {"agree": pred[0], 
                "disagree": pred[1],
                "discuss" : pred[2],
                "unrelated" : pred[3]}
            resource_usage={'memory used':memory_used,'cpu used':cpu_used,'network used':net_used,'time_taken':time_taken}
            
            txn_hash=telemetry.call_telemetry(str(result),cpu_used,memory_used,net_used,time_taken,call_id)
            response=[str(result),str(txn_hash),str(resource_usage)]
            response=str(response)
            current_span.add_event("event message", {"result": str(response)})
            logger.info(response)
        except Exception as e:
            exception = Exception(str(e))
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, "error happened"))
            logger.error(e)
        resp.response=response
        logger.info(str(resp.response))  
        logger.info(str(stance_pred))  
        return resp

class GRPCproto(service_proto_pb2_grpc.ProtoDefnitionServicer):
    def req_service_price(self, req, ctxt):
        priceParams=service_proto_pb2.priceRespService()
        priceParams.cost_per_process=75
        priceParams.pubk="0xb5114121A51c6FfA04dBC73F26eDb7B6bfE2eB35"
        return priceParams

    def req_msg(self, req, ctxt):
        #TODO:  use https://googleapis.dev/python/protobuf/latest/ instead of reading from file 
        with open('service/service_spec/factai_service.proto', 'r') as file:
            proto_str = file.read()
        reqMessage=service_proto_pb2.reqMessage()
        reqMessage.proto_defnition=proto_str
        reqMessage.service_stub="FACTAIStanceClassificationStub"
        reqMessage.service_input="InputData"
        reqMessage.function_name="stance_classify"
        return reqMessage
    
    def req_metadata(self, req, ctxt):
        #TODO:  use https://googleapis.dev/python/protobuf/latest/ instead of reading from file 
        with open('service/service_spec/factai_service.proto', 'r') as file:
            proto_str = file.read()
        service_name=req.service_name
        
        respMetadata=service_proto_pb2.respMetadata()
        
        proto_defnition=proto_str
        service_stub="FACTAIStanceClassificationStub"
        service_input="InputData"
        function_name="stance_classify"
        
        deployment_type=os.environ['deployment_type']       
        if deployment_type=="prod":    
            with open('service/service_spec/service_definition_prod.json', 'r') as file:
                service_definition_str = file.read()
        else:
            with open('service/service_spec/service_definition.json', 'r') as file:
                service_definition_str = file.read()
        
        service_definition=json.loads(service_definition_str)
        service_definition["declarations"]["protobuf_definition"]=proto_defnition
        service_definition["declarations"]["service_stub"]=service_stub
        service_definition["declarations"]["function"]=function_name
        service_definition["declarations"]["input"]=service_input
        respMetadata.service_definition=json.dumps(service_definition)
        return respMetadata

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


sess = tf.Session()
load_model(sess)
grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
pb2_grpc.add_FACTAIStanceClassificationServicer_to_server(GRPCapi(sess), grpc_server)
service_proto_pb2_grpc.add_ProtoDefnitionServicer_to_server(GRPCproto(), grpc_server)
grpc_server.add_insecure_port('[::]:' + str(grpc_port))
grpc_server.start()
logger.info("GRPC Server Started on port: " + str(grpc_port))
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    logger.error("Exiting....")
    grpc_server.stop(0)
