import sys
import os
import signal
import time
import subprocess
import logging
import pathlib
import glob
import json
import argparse

from service import registry

logging.basicConfig(level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger("run_factai_service")

def main():
	parser = argparse.ArgumentParser(description="Run services")
	parser.add_argument(
		"--no-daemon", 
		action="store_false", 
		dest="run_daemon", 
		help="do not start the daemon")
	parser.add_argument(
		"--daemon-config",
		help="Path of daemon configuration file, without config it won't be started",
		required=False)
	parser.add_argument(
		"--ssl",
		action="store_true",
		dest="run_ssl",
		help="start the daemon with SSL"
		)

	args = parser.parse_args()
	root_path = pathlib.Path(__file__).absolute().parent

	# All services modules go here
	service_modules = ["service.factai_service"]

	# call for all the services listed in service_modules
	all_p = start_all_services(root_path, service_modules, args.run_daemon, args.daemon_config, args.run_ssl)

	# continous checking all subprocess
	try:
		while True:
			for p in all_p:
				p.poll()
				if p.returncode and p.returncode != 0:
					kill_and_exit(all_p)
			time.sleep(1)
	except Exception as e:
		print(e)
                #log.error(e)
		raise

def start_all_services(cwd, service_modules, run_daemon, daemon_config, run_ssl):
	"""
	Loop through all service_modules and start them.
	For each one, an instance of Daemon "snetd" is created
	snetd will start with configs from "snetd."
	"""
	all_p = []
	for i, service_module in enumerate(service_modules):
		service_name = service_module.split(".")[-1]
		#log.info("Launching {} on port {}".format(str(registry[service_name]), service_module))
		all_p += start_service(cwd, service_module, run_daemon, daemon_config, run_ssl)

	return all_p

def start_service(cwd, service_module, run_daemon, daemon_config, run_ssl):
	"""
	Starts SNET Daemon("snetd") and the python module of the service 
	at the passed gRPC port
	"""	

	def add_ssl_configs(conf):
		""" Add SSL keys to snetd.config.json"""
		with open(conf, "r") as f:
			snetd_configs = json.load(f)
			snetd_configs["ssl_cert"] = "/opt/singnet/.certs/fullchain.pem"
			snetd_configs["ssl_key"] = "/opt/singnet/.certs/privKey.pem"

		with open(conf, "w") as f:
			json.dump(snetd_configs, f, sort_keys=True, indent=4)

	all_p = []
	if run_daemon:
		if daemon_config:
			all_p.append(start_snetd(str(cwd), daemon_config))
		else:
			for idx, config_file in enumerate(glob.glob("./snetd_configs/*.json")):
				if run_ssl:
					add_ssl_configs(config_file)
				all_p.append(start_snetd(str(cwd), config_file))

	service_name = service_module.split(".")[-1]
	grpc_port = registry[service_name]["grpc"]
	p = subprocess.Popen([sys.executable, "-m", service_module, "--grpc-port", str(grpc_port)], cwd=str(cwd))
	all_p.append(p)
	return all_p 

def start_snetd(cwd, config_file=None):
	"""
	Starts the daemon "snetd":
	"""
	cmd = ["snetd", "serve"]
	if config_file:
		cmd = ["snetd", "serve", "--config", config_file]

	return subprocess.Popen(cmd, cwd=str(cwd))

def kill_and_exit(all_p):
	for p in all_p:
		try:
			os.kill(p.pid, signal.SIGTERM)
		except Exception as e:
			print(e)
                        #log.error(e)	
	exit(1)

if __name__ == "__main__":
	main()
