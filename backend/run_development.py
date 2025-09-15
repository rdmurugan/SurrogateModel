#!/usr/bin/env python3
"""
Development server runner for the Surrogate Model Platform.

This script sets up and runs the development environment with proper
configuration for the Active Learning system.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def setup_environment():
    """Set up the development environment"""
    print("🔧 Setting up development environment...")

    # Create .env file if it doesn't exist
    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from .env.example...")
        import shutil
        shutil.copy(env_example, env_file)
        print("   ✅ .env file created")
        print("   ⚠️  Please review and update the .env file with your configuration")

    # Create necessary directories
    directories = [
        "logs",
        "uploads",
        "models",
        "data"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    print("   ✅ Directories created")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("📦 Checking dependencies...")

    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import numpy
        import sklearn
        print("   ✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        print("   💡 Run: pip install -r requirements.txt")
        return False

def init_database():
    """Initialize the database"""
    print("🗄️  Initializing database...")

    try:
        # Import after ensuring dependencies are available
        from app.core.database import init_db
        init_db()
        print("   ✅ Database initialized")
        return True
    except Exception as e:
        print(f"   ⚠️  Database initialization warning: {e}")
        return True  # Continue anyway for development

def run_server():
    """Run the development server"""
    print("🚀 Starting development server...")
    print("=" * 50)

    # Set environment variables for development
    os.environ.setdefault("DEBUG", "true")
    os.environ.setdefault("DATABASE_URL", "sqlite:///./surrogate_platform.db")

    try:
        # Run uvicorn server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info"
        ]

        print("📡 Server will be available at:")
        print("   • API: http://localhost:8000")
        print("   • Docs: http://localhost:8000/docs")
        print("   • Interactive API: http://localhost:8000/redoc")
        print("")
        print("🧪 Test the Active Learning API with:")
        print("   python test_api.py")
        print("")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 50)

        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

def main():
    """Main function"""
    print("🏗️  Surrogate Model Platform - Development Setup")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("app").exists():
        print("❌ Please run this script from the backend directory")
        print("   cd backend && python run_development.py")
        sys.exit(1)

    # Setup steps
    setup_environment()

    if not check_dependencies():
        print("\n💡 To install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n   Or with virtual environment:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    init_database()

    print("\n✅ Setup complete! Starting server...")
    time.sleep(1)

    run_server()

if __name__ == "__main__":
    main()