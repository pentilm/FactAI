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

    def call_telemetry(self,cpu_used,memory_used,net_used,time_taken):
        channel = grpc.insecure_channel("{}".format("demo.huggingface.io:7003"))
        stub = telemetry_pb2_grpc.HuggingfaceAdapterStub(channel)
        result=stub.telemetry(telemetry_pb2.TelemetryInput(cpu_used=cpu_used,memory_used=memory_used,net_used=net_used,time_taken=time_taken,device_name=self.device_name))
        return str(result)
