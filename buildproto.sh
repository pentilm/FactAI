#! /bin/bash
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/service_spec/factai_service.proto

python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/service_spec/service_proto.proto

python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/service_spec/telemetry.proto

python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/service_spec/registry.proto
