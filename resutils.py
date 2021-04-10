from resource import *
import json
import time
import requests
import os
import subprocess
import sys

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
        params='{"device_name":"'+self.device_name+'","cpu_used": "'+cpu_used+'","memory_used":"'+memory_used+'","net_used":"'+net_used+'","time_taken":"'+time_taken+'"}'
        subprocess.Popen(["grpcurl", "-plaintext", "-proto", "service/service_spec/telemetry.proto", "-d",  str(params) , "195.201.197.25:50000" , "session_manager.SessionManager/telemetry"],stderr=subprocess.PIPE)
