# AgriSmart - Precision Agriculture Platform

## Overview

AgriSmart is a comprehensive precision agriculture platform that enables intelligent pesticide management through AI-powered drone integration. The system helps farmers optimize crop health monitoring, reduce pesticide waste, and increase yields through smart agricultural automation. The application provides real-time field mapping, plant health analysis, and automated pesticide application recommendations based on drone-collected data.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

The frontend is built as a React Single Page Application (SPA) with modern tooling:

- **Framework**: React 18 with TypeScript for type safety and developer experience
- **Styling**: Tailwind CSS with shadcn/ui components for consistent, modern UI design
- **Routing**: Wouter for lightweight client-side routing
- **State Management**: TanStack Query for server state management and caching
- **Form Handling**: React Hook Form with Zod validation for robust form management
- **Build Tool**: Vite for fast development and optimized production builds

The UI follows a dashboard-based design with protected routes requiring authentication. The application uses a sidebar navigation pattern with responsive mobile support.

### Backend Architecture

The backend is a REST API built with Node.js and Express:

- **Runtime**: Node.js with TypeScript and ES modules
- **Framework**: Express.js for HTTP server and API routing
- **Authentication**: JWT-based authentication with bcrypt for password hashing
- **Storage**: PostgreSQL database with Drizzle ORM for type-safe database operations
- **Session Management**: Session-based authentication with secure JWT tokens

The API follows RESTful conventions with proper error handling and request validation using Zod schemas.

### Data Storage Solutions

**Database**: PostgreSQL with Drizzle ORM
- **Users**: Authentication and profile management
- **Crops**: Crop type definitions, planting schedules, and growth stages
- **Fields**: Geographic field boundaries and crop assignments
- **Drone Connections**: Device management and connection status
- **Plant Health Records**: AI-generated health assessments and infection data
- **Pesticide Applications**: Treatment recommendations and application tracking
- **Contact Messages**: Customer support and communication

The database schema is designed for agricultural workflows with proper foreign key relationships and data integrity constraints.

### Authentication and Authorization

- **JWT Tokens**: Stateless authentication with 7-day expiration
- **Password Security**: Bcrypt hashing with salt rounds for secure password storage
- **Route Protection**: Middleware-based authentication checking for protected endpoints
- **Role-Based Access**: User roles system with farmer as default role
- **Session Management**: Client-side token storage with automatic refresh handling

### External Service Integrations

**Neon Database**: Serverless PostgreSQL hosting for scalable data storage
**Mapping Services**: Integration points for field mapping and GPS coordinate handling
**Drone Communication**: Bluetooth and WiFi protocols for drone connectivity
**Real-time Data**: WebSocket/MQTT architecture prepared for live drone data streaming

The system is designed with modularity to easily integrate additional agricultural sensors, weather APIs, and third-party farming management systems.

## External Dependencies

- **@neondatabase/serverless**: PostgreSQL database connection and serverless deployment
- **drizzle-orm**: Type-safe database ORM with PostgreSQL dialect
- **@tanstack/react-query**: Server state management and API caching
- **@radix-ui**: Accessible UI component primitives for forms and interactions
- **bcrypt**: Password hashing and security
- **jsonwebtoken**: JWT token generation and verification
- **react-hook-form**: Form validation and submission handling
- **zod**: Runtime type validation and schema definition
- **tailwindcss**: Utility-first CSS framework for styling
- **wouter**: Lightweight React router for SPA navigation