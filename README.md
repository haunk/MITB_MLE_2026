# CS611 - Machine Learning Engineering

> Singapore Management University (SMU) | School of Computing
> Master of IT in Business - Artificial Intelligence

## Course Overview

This course bridges the gap between building ML models and deploying them in production. While Applied Machine Learning (CS610) teaches how to build models, MLE focuses on creating maximum value by deploying, operationalizing, and maintaining ML systems with engineering excellence.

The course uses **open-source, cloud-compatible, cloud-agnostic tools** including Airflow, PySpark, Python, Jupyter Notebooks, Docker, and more.

## Learning Objectives

- Understand the definition and importance of machine learning engineering and solutions architecting
- Build ML pipelines covering data preprocessing, model training, evaluation, deployment, monitoring, and more
- Deploy ML pipelines through CI/CD (Continuous Integration / Continuous Deployment) practices
- Deploy ML pipelines and CI/CD systems on cloud infrastructure
- Adopt and apply machine learning engineering best practices in real-world settings

## Course Topics

### Lecture 1: Introduction

- **Machine Learning Lifecycle** - End-to-end view from data to deployment
- **Challenges of Operationalizing ML** - Training-serving skew, experiment tracking, continuous retraining, model fairness and ethics
- **Requirements for Production ML Systems** - Reliability, Scalability, Maintainability, Adaptability, Iterative Process
- **ML Pipeline and ML Steps** - Data processing, model training, model evaluation, model deployment, pipeline orchestration
- **MLOps** - Methodology unifying ML development with ML operations (analogous to DevOps for software engineering)

### Key Concepts Covered Throughout the Course

| Area | Topics |
|------|--------|
| Data Processing | ETL flows, data validation, schema validation, training/serving skew detection |
| Model Training | Algorithm selection (ML, DL, GenAI), training requirements, experiment tracking |
| Model Evaluation | Performance metrics, value thresholds, change thresholds, baseline comparison |
| Model Deployment | Batch vs. real-time inference, canary testing, A/B testing, rollback planning |
| Pipeline Orchestration | Task dependency graphs, triggering strategies (on-demand, scheduled, data-driven) |
| Model Monitoring | Data drift detection, performance degradation, uptime monitoring |
| MLOps | CI/CD for ML, environment management, governance, compliance |

## Production ML System Requirements

1. **Reliability** - System performs correctly even under adversity (hardware/software faults, human error). ML systems can fail silently, unlike traditional software.
2. **Scalability** - Handle growth in model complexity, traffic volume, and model count. Autoscaling and artifact/metadata management.
3. **Maintainability** - Support collaboration across ML engineers, DevOps engineers, data analysts, and data scientists working with diverse file types (.py, .sql, .ipynb, .yaml, .sh, etc.).
4. **Adaptability** - Evolve quickly as business requirements and data distributions shift. In ML systems, behavior is defined by data, not just code.

## Tools and Technologies

- **Languages:** Python 3, PySpark, Shell scripting
- **Pipeline Orchestration:** Apache Airflow
- **Compute:** Cluster computing (PySpark), CPU/GPU
- **Containerization:** Docker
- **Notebooks:** Jupyter
- **Cloud:** Cloud-agnostic design principles


### Project Breakdown

The course project requires building an end-to-end ML pipeline using open-source tools:

- Select your own dataset and ML model
- Build a complete ML pipeline (data processing, model training, deployment/serving)
- Document design architecture and justify design choices
- Provide a README with steps to run the pipeline
- CI/CD is optional but recommended


---
*Deeply thank you Professor Ulysses Chong Min Zhen for guiding us.*
