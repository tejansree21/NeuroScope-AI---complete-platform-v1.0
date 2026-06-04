
"""
NeuroScope AI - Federated Learning Client
Run this at each hospital:
  python fl_client.py --hospital-id A --server neuroscope-fl-server:8080
"""

import argparse
import flwr as fl
import torch
from neuroscope_fl import NeuroscopeFlowerClient, load_local_data

parser = argparse.ArgumentParser()
parser.add_argument('--hospital-id',  required=True)
parser.add_argument('--server',       default='localhost:8080')
parser.add_argument('--dp',           action='store_true', help='Enable differential privacy')
parser.add_argument('--data-path',    required=True, help='Path to local DICOM data')
args = parser.parse_args()

# Load local data (never leaves this machine)
dataset = load_local_data(args.data_path)
model   = torch.load('backbone.pth')   # shared backbone (not patient data)

client = NeuroscopeFlowerClient(
    hospital_id=args.hospital_id,
    dataset=dataset,
    model=model,
    dp_enabled=args.dp,
)

# Connect to FL server and start training
fl.client.start_numpy_client(
    server_address=args.server,
    client=client,
)
