# 🚀 Surrogate Model Platform

> **Advanced Active Learning & Adaptive Sampling for Engineering Simulations**

A comprehensive cloud-based SaaS platform that reduces simulation costs by **70-80%** through intelligent data collection and surrogate modeling. Features cutting-edge active learning algorithms, physics-informed sampling, and multi-fidelity modeling for engineering applications.

## 🎯 Key Features

- **🧠 Intelligent Active Learning**: Adaptive sampling strategies with physics-informed constraints
- **⚡ Multi-Fidelity Modeling**: Combines low and high-fidelity simulations efficiently
- **🔄 Real-Time Adaptation**: Dynamic strategy switching based on model performance
- **🛡️ Enterprise Security**: JWT authentication with multi-tenant isolation
- **📊 Advanced Analytics**: Uncertainty quantification and convergence monitoring
- **🚀 Scalable Architecture**: Async processing with Docker containerization
- **🎯 Smart Acquisition**: Expected Improvement, UCB, Knowledge Gradient functions
- **⚙️ Physics Integration**: Domain knowledge and conservation laws
- **📈 Budget Optimization**: Cost-aware sampling with resource tracking

## 🏗️ Architecture

### Backend
- **FastAPI**: High-performance Python web framework
- **PostgreSQL**: Primary database for metadata
- **Redis**: Caching and session management
- **MinIO**: S3-compatible object storage for datasets
- **SQLAlchemy**: Database ORM with Alembic migrations
- **Celery**: Background task processing

### Frontend
- **React 18**: Modern JavaScript framework
- **TypeScript**: Type-safe development
- **Material-UI**: Professional UI components
- **Plotly.js**: Interactive data visualizations
- **Axios**: HTTP client for API communication

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Local development orchestration
- **Nginx**: Reverse proxy and static file serving

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend development)

### Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SurrogateModel
   ```

2. **Start all services**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - MinIO Console: http://localhost:9001

### Local Development Setup

#### Backend Setup
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Run database migrations
alembic upgrade head

# Start the development server
uvicorn app.main:app --reload
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

## 📝 Usage

### 1. Upload Dataset
- Navigate to the Datasets page
- Click "Upload Dataset"
- Provide CSV/Excel file with simulation results
- Specify input and output columns

### 2. Train Surrogate Model
- Go to Models page
- Click "Create Model"
- Select your dataset and algorithm
- Wait for training to complete

### 3. Make Predictions
- Visit Predictions page
- Select your trained model
- Enter input parameters
- Get instant predictions with uncertainty quantification

### 4. API Integration
```python
import requests

# Login
response = requests.post("http://localhost:8000/api/v1/auth/login", data={
    "username": "your_email",
    "password": "your_password"
})
token = response.json()["access_token"]

# Make prediction
headers = {"Authorization": f"Bearer {token}"}
prediction_data = {"length": 10, "width": 5, "thickness": 2}

response = requests.post(
    "http://localhost:8000/api/v1/predictions/1/predict",
    json=prediction_data,
    headers=headers
)
result = response.json()
```

## 🔧 Development

### Backend Commands
```bash
cd backend

# Run tests
make test

# Run tests with coverage
make test-cov

# Lint code
make lint

# Format code
make format

# Create database migration
make migrate-create name="add_new_table"

# Apply migrations
make migrate
```

### Frontend Commands
```bash
cd frontend

# Run tests
npm test

# Build for production
npm run build

# Type checking
npm run type-check
```

## 🏗️ Project Structure

```
SurrogateModel/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core functionality
│   │   ├── db/             # Database configuration
│   │   ├── models/         # SQLAlchemy models
│   │   └── services/       # Business logic
│   ├── tests/              # Backend tests
│   └── alembic/            # Database migrations
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   └── types/          # TypeScript types
│   └── public/             # Static assets
├── infrastructure/         # Deployment configs
└── docs/                   # Documentation
```

## 🔐 Security

- JWT-based authentication
- Role-based access control (RBAC)
- Tenant isolation
- Input validation and sanitization
- SQL injection prevention
- CORS configuration
- Secure password hashing

## 📊 Monitoring

- Health check endpoints
- Structured logging
- Performance metrics
- Error tracking
- API response time monitoring

## 🚀 Deployment

### Production Deployment
1. Configure environment variables
2. Set up external databases (PostgreSQL, Redis)
3. Configure object storage (S3/MinIO)
4. Deploy with Kubernetes or Docker Swarm
5. Set up monitoring and logging
6. Configure SSL/TLS certificates

### Environment Variables
See `backend/.env.example` for all available configuration options.

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📈 Performance

- Sub-100ms prediction response times
- Horizontal scaling support
- Efficient database queries
- Caching for frequently accessed data
- Optimized ML model serving

## 🤝 Contributing

We welcome contributions from the engineering and ML communities! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**Quick Start:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run tests: `pytest tests/ && python test_api.py`
5. Submit a pull request

**Priority Areas:**
- 🧠 Advanced acquisition functions and sampling strategies
- ⚡ Performance optimizations and parallelization
- 🔧 Integration with commercial simulation tools (ANSYS, COMSOL, OpenFOAM)
- 📊 Advanced visualization and analysis tools
- 🏭 Industry-specific examples and benchmarks

**Research Contributions:** We encourage academic contributions! Please reference original papers and include proper validation against benchmarks.

## 📄 License

This project is licensed under a **Custom License** for personal and research use:

- ✅ **Free for personal use** and research
- ✅ **Open source contributions** welcome
- ✅ **Academic and educational** use permitted
- 🏢 **Commercial use** requires separate license

For commercial licensing, enterprise support, or custom development:
**📧 Contact: durai@infinidatum.net**

See the [LICENSE](LICENSE) file for complete terms and conditions.

## 🆘 Support & Contact

- **📚 Documentation**: See `/docs` folder and http://localhost:8000/docs
- **🐛 Issues**: GitHub Issues for bugs and feature requests
- **💼 Commercial Licensing**: durai@infinidatum.net
- **🤝 Enterprise Support**: durai@infinidatum.net
- **🔧 Custom Development**: durai@infinidatum.net

For any questions about licensing, commercial use, or custom implementations, please reach out to **durai@infinidatum.net**.