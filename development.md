# development.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains the design and implementation for a cloud-based SaaS platform for surrogate modeling in engineering. The platform is designed to help engineering companies reduce computational costs and accelerate design cycles by replacing expensive FEA/CFD simulations with fast, accurate surrogate models.

## Project Architecture

**Platform Type**: Cloud-native SaaS application for engineering surrogate modeling

**Key Documentation**:
- `PLATFORM_DESIGN.md` - Comprehensive platform design including architecture, features, and business model
- `README.md` - Project overview

## Technology Stack (Planned)

**Backend**:
- FastAPI (Python) for API services
- PyTorch/TensorFlow for ML models
- PostgreSQL + Redis for data management
- Celery for task processing

**Frontend**:
- React with TypeScript
- Plotly.js for data visualization

**Infrastructure**:
- Kubernetes on AWS/GCP/Azure
- Docker containerization
- S3-compatible storage for simulation data

## Development Workflow

When implementing features:

1. **Follow the architecture** outlined in PLATFORM_DESIGN.md
2. **Multi-tenant design** - ensure all code supports tenant isolation
3. **Security-first** - implement proper authentication, authorization, and data encryption
4. **ML best practices** - use proper validation, model versioning, and monitoring
5. **API-first development** - ensure all features are accessible via REST API

## Key Development Considerations

- **Data Privacy**: Handle sensitive engineering IP with appropriate security measures
- **Scalability**: Design for horizontal scaling and high-throughput predictions
- **ML Operations**: Implement proper model lifecycle management
- **Performance**: Optimize for sub-second prediction response times
- **Compliance**: Consider ITAR/EAR regulations for engineering data

## Common Commands

**Backend Development:**
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload

# Run tests
make test

# Run tests with coverage
make test-cov

# Lint and format code
make lint
make format

# Database migrations
make migrate                    # Apply migrations
make migrate-create name="..."  # Create new migration
```

**Frontend Development:**
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build
```

**Docker Commands:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild containers
docker-compose build
```

## Project Structure

When code is added, organize as:
```
/backend          # FastAPI services
/frontend         # React application
/ml-models        # Surrogate model implementations
/infrastructure   # Kubernetes manifests, Dockerfiles
/docs            # Additional documentation
```
