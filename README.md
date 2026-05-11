# 💰 NextGen Finance – AI-Powered Personal Finance Management System

NextGen Finance is a full-stack AI-powered personal finance management platform that helps users track income and expenses, set budgets, import CSV transaction data, and receive machine learning-based financial insights such as overspending risk and next-month expense predictions.

---

## 🚀 Features

### 👤 User Authentication
- Secure user registration and login
- Password hashing using bcrypt
- JWT-based authentication
- Protected routes

### 💸 Transaction Management
- Add, edit, delete, and view transactions
- Categorize income and expenses
- Search and filter transactions

### 📊 Budget Management
- Create monthly budgets
- Track spending against budgets
- Budget alerts and summaries

### 📁 CSV Import
- Upload transaction CSV files
- Preview and validate data before importing
- Normalize data from different bank formats

### 🤖 Machine Learning Insights
- Overspending risk prediction using Logistic Regression
- Next month expense forecasting using Linear Regression
- Financial behavior clustering using K-Means
- Anomaly detection and confidence scoring

### 🧠 AI Financial Advisor
- Personalized financial recommendations
- Spending analysis and suggestions

### 🔔 Real-Time Notifications
- Instant notifications using Socket.IO
- Budget warnings and transaction alerts

### 📈 Analytics Dashboard
- Expense summaries
- Category breakdowns
- Trends and reports

---

## 🛠️ Tech Stack

### Frontend
- React.js
- Vite
- Tailwind CSS
- Context API
- Axios
- Socket.IO Client

### Backend
- Node.js
- Express.js
- MongoDB
- Mongoose
- JWT
- bcrypt
- Multer

### Machine Learning
- Python
- scikit-learn
- Pandas
- NumPy

### Deployment
- Vercel (Frontend)
- Render (Backend)

---

## 🧠 Algorithms Used

| Algorithm | Purpose |
|--------|--------|
| Logistic Regression | Predict overspending risk |
| Linear Regression | Forecast next month expenses |
| K-Means Clustering | Group spending behavior patterns |
| StandardScaler | Normalize data before prediction |
| Statistical Anomaly Detection | Detect unusual spending |
| bcrypt | Secure password hashing |
| JWT | Authentication and authorization |

---

## 🏗️ System Architecture

```text
React Frontend
      ↓
Express REST API
      ↓
Authentication Middleware (JWT)
      ↓
Controllers
      ↓
Services
      ↓
MongoDB Database
      ↓
Python ML Pipeline
      ↓
Insights & Predictions
      ↓
Socket.IO Notifications
