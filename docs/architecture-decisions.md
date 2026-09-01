# Architecture Decision Records

## ADR-001: Use Amazon SageMaker as the Primary ML Platform

### Status

Accepted

### Context

The project requires capabilities spanning data preparation, model training, hyperparameter optimization, model management, deployment, monitoring, and automated ML workflows.

Building these capabilities independently would introduce unnecessary operational complexity.

### Decision

Amazon SageMaker will serve as the primary managed ML platform.

### Rationale

SageMaker provides integrated capabilities for:

- Data processing
- Model training
- Hyperparameter tuning
- Feature management
- Model registration
- Model deployment
- Model monitoring
- ML workflow orchestration

This allows the project to demonstrate the complete ML lifecycle while minimizing undifferentiated infrastructure management.

### Alternatives Considered

#### Self-managed ML infrastructure

Rejected for the primary implementation because it would require managing training infrastructure, model serving infrastructure, scaling, monitoring, and associated operational components.

#### Amazon EC2

Rejected as the primary ML platform because EC2 provides compute infrastructure but does not provide the same integrated ML lifecycle capabilities.

### Consequences

Positive:

- Reduced infrastructure management
- Native AWS integration
- Easier ML lifecycle orchestration
- Strong alignment with the MLA-C01 objectives

Negative:

- AWS service complexity
- Potential vendor lock-in
- SageMaker-specific operational knowledge required
- Potential cost if resources are left running



## ADR-002: Use XGBoost as the Primary Production Candidate

### Status

Proposed

### Context

The primary dataset is structured/tabular customer data.

### Decision

XGBoost will be evaluated as the primary production candidate, with a simpler baseline model established first.

### Rationale

XGBoost is well suited to structured/tabular classification problems and supports nonlinear relationships and feature interactions.

The decision will ultimately be validated using model evaluation results.

### Consequences

Positive:

- Strong candidate for tabular data
- Mature algorithm
- Good performance on many structured-data problems
- Supported by SageMaker

Negative:

- More complex than a linear baseline
- Model interpretability requires additional tooling
- Hyperparameter tuning can increase training cost
