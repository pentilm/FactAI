from resource import *
import json
import time
import requests
import os
import subprocess
import sys
import grpc
import service.service_spec.telemetry_pb2 as telemetry_pb2
import service.service_spec.telemetry_pb2_grpc as telemetry_pb2_grpc

import service.service_spec.registry_pb2_grpc as registry_pb2_grpc
import service.service_spec.registry_pb2 as registry_pb2

class resutils():
    def __init__(self):
        self.device_name="factai"


    def memory_usage(self):
        to_MB = 1024.
        to_MB *= to_MB
        return getrusage(RUSAGE_SELF).ru_maxrss / to_MB

    def cpu_ticks(self):
        sec= time.clock()
        tick=sec*(10**7)
        mtick=tick/10**6
        return mtick

    def block_in(self):
        to_KB = 1024.
        return getrusage(RUSAGE_SELF).ru_inblock/to_KB

    def get_address(self):
        service_name=os.environ['huggingface_adapter_name']       
        service_address="135.181.222.170"
        if service_name=="testing-huggingface-adapter-factai":    
            service_port=60778
        else:
            service_port=60777
        return str(service_address),str(service_port)

    def call_telemetry(self,stance_pred,cpu_used,memory_used,net_used,time_taken):
        req_address=self.get_address()
        address=req_address[0]
        port=req_address[1]
        huggingface_adapter_address=address+":"+port
        channel = grpc.insecure_channel("{}".format(huggingface_adapter_address))
        stub = telemetry_pb2_grpc.HuggingfaceAdapterStub(channel)
        result=stub.telemetry(telemetry_pb2.TelemetryInput(result=stance_pred,cpu_used=cpu_used,memory_used=memory_used,net_used=net_used,time_taken=time_taken,device_name=self.device_name))
        return str(result.response)
