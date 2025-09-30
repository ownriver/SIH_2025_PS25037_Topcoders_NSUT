# Tasks Accomplished

- **Task 1: Frontend Development**
  - Built a responsive React SPA for precision agriculture, including dashboard, crop selection, drone connection, and contact pages.
- **Task 2: Backend API Implementation**
  - Developed RESTful API using Node.js and Express for user authentication, crop data, drone metrics, and plant health records.
- **Task 3: Database Integration**
  - Integrated PostgreSQL with Drizzle ORM for secure, type-safe data storage and retrieval.

# Technology Stack
This project leverages the following technologies:

- **React + TypeScript**: Chosen for fast development, type safety, and building a modern, responsive SPA dashboard.
- **Tailwind CSS + shadcn/ui**: Used for clean, consistent, and highly customizable UI components.
- **Node.js + Express**: Powers the backend API, enabling scalable and efficient server-side logic.
- **PostgreSQL + Drizzle ORM**: Ensures reliable, type-safe, and relational data management for all agricultural workflows.
- **TanStack Query**: Manages server state and caching for seamless frontend-backend data sync.
- **JWT + bcrypt**: Provides secure authentication and password management.
- **React Hook Form + Zod**: Enables robust form validation and management.
- **Wouter**: Lightweight routing for SPA navigation.

# Key Features
- **Precision Spraying**: Advanced GPS and sensor tech ensure fertilizer is applied exactly where needed, minimizing waste.
- **Real-time Monitoring**: Monitor crop health, soil conditions, and application progress via the dashboard.
- **Smart Application**: AI algorithms analyze field data to optimize fertilizer mix and timing.

# Local Setup Instructions
Follow these steps to run the project locally:

## 1. Clone the Repository
```sh
git clone https://github.com/ownriver/SIH_2025_Internal_Round_Submission_PS25015.git
cd SIH_2025_Internal_Round_Submission_PS25015/codes/codes/website
```

## 2. Install Dependencies
### Windows
```powershell
cd client; npm install; cd ..
cd server; npm install; cd ..
```
### macOS/Linux
```bash
cd client && npm install && cd ..
cd server && npm install && cd ..
```

## 3. Set Up Environment Variables
- Create a `.env` file in `server/` and add your PostgreSQL `DATABASE_URL` and JWT secret.

## 4. Run the Development Servers
### Windows
```powershell
cd client; npm run dev; cd ..
cd server; npm run dev; cd ..
```
### macOS/Linux
```bash
cd client && npm run dev &
cd server && npm run dev &
```

## 5. Access the Application
- Open [http://localhost:5173](http://localhost:5173) for the frontend.
- Backend API runs on [http://localhost:3000](http://localhost:3000).

---
For more details, see the main README and `replit.md` in the project root.