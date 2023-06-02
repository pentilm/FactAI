from service.util import *
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
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
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags


jaeger_address = os.environ['jaeger_address']


jaeger_exporter = JaegerExporter(
    agent_host_name=jaeger_address,
    agent_port=6831,
)


trace_provider=trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# grpc_server_instrumentor = GrpcInstrumentorServer()
# grpc_server_instrumentor.instrument()

# tracer_server=opentelemetry.instrumentation.grpc.server_interceptor(tracer_provider=trace_provider)
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
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
feature_size = len(train_set[0])
test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)



def train_save():
    X = np.asarray(train_set)[:500]
    y = np.asarray(train_stances)[:500]



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # # determine the number of input features
    n_features = X_train.shape[1]

    print(n_features)
    model = Sequential()
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(1001,)))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(4, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the model
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
    # evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)

    model.save('saved_model/factai_model')

# TRAIN THE MODEL IF YOU WANT TO 
# train_save()

# HOW TO DO PREDICTION INSIDE THE PIPELINE
model = tf.keras.models.load_model('saved_model/factai_model')


input_data = {'headline' : "headline",'body' : "body"}
test_set = pipeline_serve(input_data,bow_vectorizer,tfreq_vectorizer,tfidf_vectorizer)
yhat = model.predict([test_set[0].tolist()])
print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))

class GRPCapi(pb2_grpc.FACTAIStanceClassificationServicer):
    def __init__(self,model):
        self.model = model

    def stance_classify(self, req, ctxt):
        current_span = trace.get_current_span()
        current_span.set_attribute("http.route", "some_route")
        sleep(30 / 1000)

        trace_info = eval(req.tracer_info)
        span_id = trace_info['span_id']
        trace_id = trace_info['trace_id']

        span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(0x01)
        )
        ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
        with tracer.start_as_current_span("stance_classify", context=ctx) as span:
            
            tracer_info=span.get_span_context()
            

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

            # test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
            # pred = self.tf_session.run(softmaxed_logits, feed_dict=test_feed_dict)[0]


            yhat = self.model.predict([test_set[0].tolist()])
            # print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

            pred = yhat.tolist()
            logger.info(pred)
            pred = pred[0]


            stance_pred = pb2.Stance()
            resp=pb2.Resp()
            stance_pred.agree = pred[0]
            stance_pred.disagree = pred[1]
            stance_pred.discuss = pred[2]
            stance_pred.unrelated = pred[3]
            response=""
            span.add_event("event message", {"prediction": str(pred)})
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
                
                txn_hash=telemetry.call_telemetry(str(result),cpu_used,memory_used,net_used,time_taken,call_id,tracer_info)
                response=[str(result),str(txn_hash),str(resource_usage)]
                response=str(response)
                #current_span.add_event("event message", {"result": str(response)})
                logger.info(response)
            except Exception as e:
                exception = Exception(str(e))
                #span.record_exception(exception)
                #span.set_status(Status(StatusCode.ERROR, "error happened"))
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
        reqMessage.service_input_params='["body","headline","call_id","tracer_info"]'
        return reqMessage
    
    def req_metadata(self, req, ctxt):
        #TODO:  use https://googleapis.dev/python/protobuf/latest/ instead of reading from file 
        with open('service/service_spec/factai_service.proto', 'r') as file:
            proto_str = file.read()
        logger.info(req.tracer_info)
        tracer_info = eval(req.tracer_info)
        logger.info("after")
        trace_id = tracer_info['trace_id']
        span_id = tracer_info['span_id']

        span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(0x01)
        )
        ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
        with tracer.start_as_current_span("req_metadata", context=ctx) as span:
            
            service_name=req.service_name
            
            respMetadata=service_proto_pb2.respMetadata()
            
            proto_defnition=proto_str
            service_stub="FACTAIStanceClassificationStub"
            service_input="InputData"
            function_name="stance_classify"
            service_input_params='["body","headline","call_id","tracer_info"]'

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
            service_definition["declarations"]["service_input_params"]=service_input_params
            respMetadata.service_definition=json.dumps(service_definition)
            return respMetadata

def run_server():
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


                    # test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
                    # pred = tf_session.run(softmaxed_logits, feed_dict=test_feed_dict)[0]

                    yhat = model.predict([test_set[0].tolist()])

                    labeled_pred = ["agree", "disagree", "discuss", "unrelated"][np.argmax(yhat)]


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


grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
pb2_grpc.add_FACTAIStanceClassificationServicer_to_server(GRPCapi(model), grpc_server)
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
