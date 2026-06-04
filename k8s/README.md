# NeuroScope AI - Kubernetes Deployment Guide

## Prerequisites
- Kubernetes cluster (local: Docker Desktop, minikube, or kind)
- Helm 3.x installed
- kubectl configured
- Docker image built: `neuroscope-ai:gpu` or `neuroscope-ai:cpu`

---

## Quick Start (Local - Docker Desktop Kubernetes)

### 1. Enable Kubernetes in Docker Desktop
Settings → Kubernetes → Enable Kubernetes → Apply

### 2. Verify cluster
```bash
kubectl cluster-info
kubectl get nodes
```

### 3. Install the Helm chart

**Tier 1 -- GPU:**
```bash
helm install neuroscope-ai ./neuroscope-ai \
  --set tier=gpu \
  --set image.tag=gpu \
  --set secrets.secretKey=your-secret-key-here \
  --set secrets.anthropicApiKey=sk-ant-your-key
```

**Tier 2 -- CPU:**
```bash
helm install neuroscope-ai ./neuroscope-ai \
  --set tier=cpu \
  --set image.tag=cpu \
  --set gpu.enabled=false
```

**Tier 3 -- Edge:**
```bash
helm install neuroscope-ai ./neuroscope-ai \
  --set tier=edge \
  --set image.tag=edge \
  --set gpu.enabled=false \
  --set redis.enabled=false \
  --set ingress.enabled=false
```

### 4. Check deployment
```bash
kubectl get pods
kubectl get services
kubectl get ingress
```

### 5. Access the API
```bash
# Port-forward if no ingress
kubectl port-forward svc/neuroscope-ai-api 8000:8000

# Then open:
# http://localhost:8000/ui
# http://localhost:8000/docs
```

---

## Production Deployment (Cloud)

### AWS EKS
```bash
# 1. Create EKS cluster with GPU node group
eksctl create cluster \
  --name neuroscope \
  --node-type p3.2xlarge \
  --nodes 2

# 2. Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml

# 3. Create models PVC from S3
# Upload models to S3 first, then use AWS EFS for ReadWriteMany

# 4. Deploy
helm install neuroscope-ai ./neuroscope-ai \
  --set tier=gpu \
  --set image.repository=<your-ecr-repo>/neuroscope-ai \
  --set image.tag=gpu \
  --set persistence.models.storageClass=efs-sc \
  --set ingress.host=neuroscope.yourdomain.com \
  --set ingress.tls.enabled=true
```

### Azure AKS
```bash
# 1. Create AKS cluster with GPU node pool
az aks create \
  --resource-group neuroscope-rg \
  --name neuroscope-cluster \
  --node-vm-size Standard_NC6s_v3 \
  --node-count 2

# 2. Deploy
helm install neuroscope-ai ./neuroscope-ai \
  --set tier=gpu \
  --set image.repository=<your-acr>.azurecr.io/neuroscope-ai
```

### GKE
```bash
# 1. Create GKE cluster with T4 GPU nodes
gcloud container clusters create neuroscope \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --machine-type n1-standard-4 \
  --num-nodes 2

# 2. Deploy
helm install neuroscope-ai ./neuroscope-ai \
  --set tier=gpu \
  --set image.repository=gcr.io/<project>/neuroscope-ai
```

---

## Upgrade
```bash
helm upgrade neuroscope-ai ./neuroscope-ai --reuse-values
```

## Rollback
```bash
helm rollback neuroscope-ai 1
```

## Uninstall
```bash
helm uninstall neuroscope-ai
# Note: PVCs are not deleted automatically -- delete manually if needed
kubectl delete pvc neuroscope-ai-models-pvc neuroscope-ai-outputs-pvc
```

---

## Multi-node Scaling

Enable HPA in values.yaml:
```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 8
  targetCPUUtilizationPercentage: 70
```

Note: GPU pods don't scale well horizontally (one GPU per pod).
For GPU scaling, use multiple node pools with separate deployments per cancer type.

---

## Monitoring

Prometheus metrics available at `/metrics` on each pod.

To enable ServiceMonitor (requires Prometheus Operator):
```yaml
monitoring:
  serviceMonitor:
    enabled: true
```

Key metrics to watch:
- `http_requests_total` -- request count per endpoint
- `http_request_duration_seconds` -- latency percentiles
- Pod CPU/memory via kubectl top pods

---

## File Structure
```
neuroscope-ai/
  Chart.yaml              -- chart metadata
  values.yaml             -- all configurable values
  templates/
    deployment.yaml       -- main API + init container
    services.yaml         -- service, ingress, PVC, Redis, HPA, secrets
    NOTES.txt             -- post-install instructions
```
