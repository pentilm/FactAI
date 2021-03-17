import argparse
import os.path
import time

from service import registry


def common_parser(script_name):
	parser = argparse.ArgumentParser(prog=script_name)
	service_name = os.path.splitext(os.path.basename(script_name))[0]
	parser.add_argument(
			"--grpc-port",
			help="port to bind gRPC service to",
			default=registry[service_name]['grpc'],
			type=int,
			required=False
		)
	return parser

# from gRPC docs:
# because start() does not block and if need to sleep-loop there is nothing
# else code to do while serving
def main_loop(grpc_handler, args):
	server = grpc_handler(port=args.grpc_port)
	server.start()
	try:
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		server.stop(0)

