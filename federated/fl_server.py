
"""
NeuroScope AI - Federated Learning Server
Run this at the coordinating institution:
  python fl_server.py --rounds 20 --min-hospitals 3
"""

import argparse
import flwr as fl

parser = argparse.ArgumentParser()
parser.add_argument('--rounds',        type=int, default=20)
parser.add_argument('--min-hospitals', type=int, default=2)
parser.add_argument('--port',          type=int, default=8080)
args = parser.parse_args()

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=args.min_hospitals,
    min_available_clients=args.min_hospitals,
)

fl.server.start_server(
    server_address=f'0.0.0.0:{args.port}',
    config=fl.server.ServerConfig(num_rounds=args.rounds),
    strategy=strategy,
)
