#! /bin/bash
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/fakenewschallenge/service_spec/factaifnc.proto
