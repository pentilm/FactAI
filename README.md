# FactAI: Intelligent Fact-Checking AI Service

Welcome to FactAI, an intelligent fact-checking service powered by advanced artificial intelligence. Developed and maintained by Zhiwei Fang, this project employs state-of-the-art algorithms designed for seamless integration with contemporary Python libraries. FactAI is designed as an AI service, optimized for on-premises platform deployments, offering a robust solution for your fact-checking needs.

## Project Overview

FactAI employs a pre-trained model to discern the relationship between the title and body of an article. It outputs an estimation of this relationship in the form of probabilities across four categories: unrelated, discussing, agreeing, and disagreeing. These probabilities sum to one, with the highest value indicating the most likely relationship as determined by our AI model.

## Deployment and Setup

FactAI leverages the power of Docker for consistent and seamless deployment. For every new commit to the `master` branch, our Continuous Integration/Continuous Deployment (CI/CD) pipeline automatically builds a Docker container. The container is then published on the Python package repository and is deployed on available hardware automatically upon a backend call from the fake-news-warning application.

To manually build and run the FactAI Docker container, follow these steps:

```sh
# Build the factai Docker image
docker build -t factai .

# Run the factai container, mapping the required ports externally
docker run -p 8020:8020 -p 8021:8021 -it factai bash

# Start the factai service
python3 run_factai_service.py --no-daemon

# Execute the test script
python3 test_factai_service.py
```

Thank you for your interest in FactAI. As the main author and developer, Zhiwei Fang believes in the power of AI to foster a more factual and informed world. We welcome your contributions and feedback as we continue to evolve and enhance this service.
