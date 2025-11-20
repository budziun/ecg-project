# 🚀 Setup Guide

Complete guide to setting up and running the ECG Arrhythmia Classifier project locally.

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git** - [Download here](https://git-scm.com/downloads)
- **Docker** - [Download here](https://www.docker.com/products/docker-desktop)
- **Docker Compose** - Included with Docker Desktop

## 🔧 Installation

### Step 1: Clone the Repository

Open your terminal and run:

git clone https://github.com/budziun/ecg-project.git


cd ecg-project


### Step 2: Build and Run with Docker

Build and start all services (backend + frontend):

docker-compose up --build

### Step 3: Access the Application

Once the containers are running, open your browser:

- **Frontend Application:** [http://localhost:3000](http://localhost:3000)
- **Backend API (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

## 🛑 Stopping the Application

### Graceful Shutdown

Press `Ctrl+C` in the terminal where Docker is running.

### Clean Shutdown

docker-compose down

### Remove All Data (Clean Reset)

docker-compose down -v

### Restart Services

docker-compose restart

## ✅ Verifying Installation

After starting the application, verify everything works:

1. ✅ Frontend loads at `http://localhost:3000`
2. ✅ API documentation loads at `http://localhost:8000/docs`
3. ✅ Upload CSV feature works
4. ✅ Model predictions display correctly
5. ✅ "About Project" modal opens and displays team information

## 📚 Additional Resources

- [Project Repository](https://github.com/budziun/ecg-project)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

## 👥 Team

- **Maciej Świder** - Project Manager, Data Scientist
- **Jakub Budzich** - ML/Web Engineer, Tech Lead
- **Adam Czaplicki** - UX Designer, QA Specialist

---

**University of Warmia and Mazury in Olsztyn • Computer Science • 2025**
