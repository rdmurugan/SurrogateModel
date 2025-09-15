# 🔧 Build and Run Guide

> **Step-by-step instructions to build and run the Surrogate Model Platform**

## 🚀 Quick Start Options

### Option 1: Local Development (Fastest ⚡)
**Best for**: Development, testing, and immediate usage

### Option 2: Docker Compose (Recommended 🐳)
**Best for**: Production-like environment with all services

### Option 3: Manual Setup (Advanced 🔧)
**Best for**: Custom configuration and debugging

---

## 🏃‍♂️ Option 1: Local Development (5 minutes)

### Prerequisites
- Python 3.11+
- Git

### Steps

```bash
# 1. Clone and navigate
git clone <repository-url>
cd SurrogateModel/backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run development server
python run_development.py
```

**That's it!** 🎉

- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Test**: `python test_api.py`

### What happens automatically:
- ✅ Creates `.env` file from template
- ✅ Sets up SQLite database
- ✅ Creates necessary directories
- ✅ Starts FastAPI server with hot reload

---

## 🐳 Option 2: Docker Compose (10 minutes)

### Prerequisites
- Docker
- Docker Compose
- Git

### Quick Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd SurrogateModel

# 2. Start all services
docker-compose up -d

# 3. Wait for health checks (30 seconds)
docker-compose ps

# 4. Test the system
cd backend && python test_api.py
```

### Services Included
- **Backend API** (Port 8000)
- **PostgreSQL** (Port 5432)
- **Redis** (Port 6379)
- **MinIO S3** (Port 9000, 9001)

### Service URLs
- **API Documentation**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001 (admin/admin)
- **Health Check**: http://localhost:8000/health

### Management Commands

```bash
# View logs
docker-compose logs -f backend

# Restart services
docker-compose restart

# Stop all services
docker-compose down

# Clean up everything
docker-compose down -v --remove-orphans
```

---

## 🔧 Option 3: Manual Setup (Advanced)

### Prerequisites
- Python 3.11+
- PostgreSQL (optional)
- Redis (optional)
- Git

### Backend Setup

```bash
# 1. Clone and setup
git clone <repository-url>
cd SurrogateModel/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Setup database (SQLite default)
# For PostgreSQL: update DATABASE_URL in .env
python -c "from app.core.database import init_db; init_db()"

# 6. Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Database Options

#### SQLite (Default - No setup required)
```bash
DATABASE_URL="sqlite:///./surrogate_platform.db"
```

#### PostgreSQL (Production recommended)
```bash
# 1. Install PostgreSQL
# 2. Create database
createdb surrogate_platform

# 3. Update .env
DATABASE_URL="postgresql://username:password@localhost:5432/surrogate_platform"
```

#### Redis (Optional - for caching)
```bash
# 1. Install Redis
# 2. Start Redis server
redis-server

# 3. Update .env
REDIS_URL="redis://localhost:6379"
```

---

## 🧪 Testing the System

### Run the Test Suite
```bash
cd backend
python test_api.py
```

### Test Output Example
```
🧪 Testing Active Learning API
==================================================
📊 Generated test data:
   - Initial samples: 10
   - Candidate points: 400

1️⃣ Creating Active Learning Session...
   ✅ Session created: abc123-def456

2️⃣ Starting Active Learning Process...
   ✅ Active learning started successfully
   📈 Max iterations: 15
   🎯 Initial samples: 10

3️⃣ Monitoring Active Learning Progress...
   🔄 Iteration 1, Samples: 13
   🔄 Iteration 2, Samples: 16
   ✅ Active learning completed!
   📊 Final results:
      - Converged: True
      - Total iterations: 8
      - Final samples: 34
      - R² score: 0.9234

🎉 Active Learning API test completed!
```

### Manual API Testing

#### 1. Check Health
```bash
curl http://localhost:8000/health
```

#### 2. View API Documentation
Open: http://localhost:8000/docs

#### 3. Test Active Learning
```python
import requests

# Create session
response = requests.post("http://localhost:8000/api/v1/active-learning/sessions",
    json={
        "model_config": {"type": "gaussian_process"},
        "sampling_config": {"adaptive": {}}
    },
    headers={"Authorization": "Bearer fake-token-for-testing"}
)
print(response.json())
```

---

## 🔍 Troubleshooting

### Common Issues & Solutions

#### ❌ Import Errors
```bash
# Solution: Install dependencies
pip install -r requirements.txt

# For development packages
pip install -e .
```

#### ❌ Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn app.main:app --port 8001
```

#### ❌ Database Connection Error
```bash
# Check DATABASE_URL in .env
# For SQLite: ensure directory exists
mkdir -p data

# For PostgreSQL: test connection
psql postgresql://username:password@localhost:5432/surrogate_platform
```

#### ❌ Permission Denied (Docker)
```bash
# Fix Docker permissions
sudo chown -R $USER:$USER .
sudo chmod -R 755 .
```

#### ❌ Module Not Found
```bash
# Ensure you're in the right directory
cd backend

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run with module flag
python -m app.main
```

### Performance Issues

#### Slow API Responses
```bash
# Check resource usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
```

#### Database Slow Queries
```bash
# Enable query logging in .env
DEBUG=true

# For PostgreSQL: check slow query log
POSTGRES_LOG_STATEMENT=all
```

---

## 🚀 Production Deployment

### Environment Preparation
```bash
# 1. Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. Create production .env
cat > .env << EOF
APP_NAME="Surrogate Model Platform"
DEBUG=false
SECRET_KEY="your-generated-secret-key"
DATABASE_URL="postgresql://user:pass@prod-db:5432/surrogate_platform"
REDIS_URL="redis://prod-redis:6379"
EOF
```

### Production Docker Setup
```bash
# 1. Build production image
docker build -t surrogate-platform:latest .

# 2. Run with production settings
docker run -d \
  --name surrogate-platform \
  -p 8000:8000 \
  --env-file .env \
  surrogate-platform:latest
```

### Health Monitoring
```bash
# Setup health checks
curl -f http://localhost:8000/health || exit 1

# Monitor logs
docker logs -f surrogate-platform

# Check metrics
curl http://localhost:8000/metrics
```

---

## 📊 Development Workflow

### Code Quality
```bash
# Format code
black app/
isort app/

# Lint code
flake8 app/

# Type checking
mypy app/
```

### Testing
```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run integration tests
python test_api.py
```

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## 🎯 Next Steps

Once the system is running:

1. **📖 Explore API Documentation**: http://localhost:8000/docs
2. **🧪 Run Test Suite**: `python test_api.py`
3. **🔧 Customize Configuration**: Edit `.env` file
4. **📊 Monitor Performance**: Check logs and metrics
5. **🚀 Deploy to Production**: Follow production guide above

### Additional Resources
- **API Reference**: `/docs` endpoint
- **Database Schema**: `app/models/` directory
- **ML Algorithms**: `app/ml/algorithms/` directory
- **Active Learning**: `app/ml/active_learning/` directory

---

**🎉 You're ready to build intelligent surrogate models with active learning!**

---

## 📄 License & Commercial Use

This software is **free for personal and research use**. For commercial applications, please contact **durai@infinidatum.net** for licensing terms.

- ✅ Personal use and research: **Free**
- ✅ Academic and educational: **Free**
- 🏢 Commercial use: **License required**

See [LICENSE](LICENSE) for complete terms.