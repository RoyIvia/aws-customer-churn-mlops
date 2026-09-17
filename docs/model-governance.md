# Model Governance

## Overview

The customer churn MLOps platform implements controlled model promotion
using automated technical validation followed by an explicit human
approval boundary.

A successfully trained model is not automatically considered suitable
for production deployment.

The promotion lifecycle is:

```text
Training
   ↓
Evaluation
   ↓
Automated Quality Gate
   ↓
SageMaker Model Registry
   ↓
PendingManualApproval
   ↓
Human Review
   ↓
Approved
   ↓
Deployment
