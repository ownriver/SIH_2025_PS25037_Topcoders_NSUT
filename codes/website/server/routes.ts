import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertUserSchema, insertCropSchema, insertFieldSchema, insertDroneConnectionSchema,
         insertPlantHealthRecordSchema, insertPesticideApplicationSchema, insertContactMessageSchema,
         loginSchema } from "@shared/schema";
import jwt from "jsonwebtoken";
import bcrypt from "bcryptjs";

const JWT_SECRET = process.env.SESSION_SECRET || 'default-secret-key';
if (!JWT_SECRET) {
  console.error('FATAL: SESSION_SECRET environment variable is required');
  process.exit(1);
}

// Middleware to verify JWT token
function authenticateToken(req: any, res: any, next: any) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.sendStatus(401);
  }

  jwt.verify(token, JWT_SECRET, (err: any, user: any) => {
    if (err) return res.sendStatus(403);
    req.user = user;
    next();
  });
}

export async function registerRoutes(app: Express): Promise<Server> {
  // Auth routes
  app.post('/api/auth/signup', async (req, res) => {
    try {
      const userData = insertUserSchema.parse(req.body);
      
      // Check if user already exists
      const existingUser = await storage.getUserByUsername(userData.username) || 
                          await storage.getUserByEmail(userData.email);
      if (existingUser) {
        return res.status(400).json({ message: "User already exists" });
      }

      // Hash password
      const hashedPassword = await bcrypt.hash(userData.password, 10);
      const user = await storage.createUser({ ...userData, password: hashedPassword });

      // Generate JWT token
      const token = jwt.sign({ userId: user.id, username: user.username }, JWT_SECRET, { expiresIn: '7d' });

      res.status(201).json({ 
        user: { id: user.id, username: user.username, email: user.email, fullName: user.fullName, role: user.role },
        token 
      });
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  app.post('/api/auth/login', async (req, res) => {
    try {
      const credentials = loginSchema.parse(req.body);
      const user = await storage.getUserByUsername(credentials.username);
      
      if (!user || !await bcrypt.compare(credentials.password, user.password)) {
        return res.status(401).json({ message: "Invalid credentials" });
      }

      const token = jwt.sign({ userId: user.id, username: user.username }, JWT_SECRET, { expiresIn: '7d' });

      res.json({ 
        user: { id: user.id, username: user.username, email: user.email, fullName: user.fullName, role: user.role },
        token 
      });
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  app.get('/api/auth/me', authenticateToken, async (req: any, res) => {
    try {
      const user = await storage.getUser(req.user.userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      res.json({ 
        user: { id: user.id, username: user.username, email: user.email, fullName: user.fullName, role: user.role }
      });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Dashboard stats
  app.get('/api/dashboard/stats', authenticateToken, async (req: any, res) => {
    try {
      const fields = await storage.getFieldsByUserId(req.user.userId);
      const healthRecords = await storage.getHealthRecordsByUserId(req.user.userId);
      const applications = await storage.getPesticideApplicationsByUserId(req.user.userId);
      
      const totalFields = fields.length;
      const totalPlants = healthRecords.length * 100; // Approximate
      const healthyPlants = Math.floor(totalPlants * 0.87);
      const averageInfectionRate = healthRecords.length > 0 
        ? healthRecords.reduce((sum, record) => sum + record.infectionRate, 0) / healthRecords.length
        : 0;
      const pesticideSaved = applications.reduce((sum, app) => sum + (app.totalVolume || 0), 0);

      res.json({
        totalFields,
        healthyPlants,
        infectionRate: averageInfectionRate,
        pesticideSaved: Math.floor(pesticideSaved)
      });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Crop routes
  app.get('/api/crops', authenticateToken, async (req: any, res) => {
    try {
      const crops = await storage.getCropsByUserId(req.user.userId);
      res.json(crops);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  app.post('/api/crops', authenticateToken, async (req: any, res) => {
    try {
      const cropData = insertCropSchema.parse({ ...req.body, userId: req.user.userId });
      const crop = await storage.createCrop(cropData);
      res.status(201).json(crop);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  app.put('/api/crops/:id', authenticateToken, async (req: any, res) => {
    try {
      // First check if crop exists and belongs to the authenticated user
      const existingCrop = await storage.getCrop(req.params.id);
      if (!existingCrop) {
        return res.status(404).json({ message: "Crop not found" });
      }
      
      if (existingCrop.userId !== req.user.userId) {
        return res.status(403).json({ message: "You can only update your own crops" });
      }

      // Validate the update data using a strict partial schema (exclude userId since it shouldn't be updated)
      const updateSchema = insertCropSchema.omit({ userId: true }).partial().strict();
      const validatedData = updateSchema.parse(req.body);
      
      const crop = await storage.updateCrop(req.params.id, validatedData);
      if (!crop) {
        return res.status(404).json({ message: "Crop not found" });
      }
      res.json(crop);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  app.delete('/api/crops/:id', authenticateToken, async (req: any, res) => {
    try {
      // First check if crop exists and belongs to the authenticated user
      const existingCrop = await storage.getCrop(req.params.id);
      if (!existingCrop) {
        return res.status(404).json({ message: "Crop not found" });
      }
      
      if (existingCrop.userId !== req.user.userId) {
        return res.status(403).json({ message: "You can only delete your own crops" });
      }

      const deleted = await storage.deleteCrop(req.params.id);
      if (!deleted) {
        return res.status(500).json({ message: "Failed to delete crop" });
      }
      
      res.status(200).json({ message: "Crop deleted successfully" });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Field routes
  app.get('/api/fields', authenticateToken, async (req: any, res) => {
    try {
      const fields = await storage.getFieldsByUserId(req.user.userId);
      res.json(fields);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  app.post('/api/fields', authenticateToken, async (req: any, res) => {
    try {
      const fieldData = insertFieldSchema.parse({ ...req.body, userId: req.user.userId });
      const field = await storage.createField(fieldData);
      res.status(201).json(field);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // Drone routes
  app.get('/api/drones', authenticateToken, async (req: any, res) => {
    try {
      const drones = await storage.getDronesByUserId(req.user.userId);
      res.json(drones);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  app.post('/api/drones', authenticateToken, async (req: any, res) => {
    try {
      const droneData = insertDroneConnectionSchema.parse({ ...req.body, userId: req.user.userId });
      const drone = await storage.createDroneConnection(droneData);
      res.status(201).json(drone);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  app.put('/api/drones/:id', authenticateToken, async (req: any, res) => {
    try {
      const drone = await storage.updateDrone(req.params.id, req.body);
      if (!drone) {
        return res.status(404).json({ message: "Drone not found" });
      }
      res.json(drone);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // Health records routes
  app.get('/api/health-records', authenticateToken, async (req: any, res) => {
    try {
      const records = await storage.getHealthRecordsByUserId(req.user.userId);
      res.json(records);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  app.post('/api/health-records', authenticateToken, async (req: any, res) => {
    try {
      const dataWithUser = { ...req.body, userId: req.user.userId };
      const recordData = insertPlantHealthRecordSchema.parse(dataWithUser);
      const record = await storage.createHealthRecord(recordData);
      res.status(201).json(record);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // Pesticide application routes
  app.get('/api/pesticide-applications', authenticateToken, async (req: any, res) => {
    try {
      const applications = await storage.getPesticideApplicationsByUserId(req.user.userId);
      res.json(applications);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  app.post('/api/pesticide-applications', authenticateToken, async (req: any, res) => {
    try {
      const dataWithUser = { ...req.body, userId: req.user.userId };
      const applicationData = insertPesticideApplicationSchema.parse(dataWithUser);
      const application = await storage.createPesticideApplication(applicationData);
      res.status(201).json(application);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  app.put('/api/pesticide-applications/:id', authenticateToken, async (req: any, res) => {
    try {
      const application = await storage.updatePesticideApplication(req.params.id, req.body);
      if (!application) {
        return res.status(404).json({ message: "Application not found" });
      }
      res.json(application);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // Contact routes
  app.post('/api/contact', async (req, res) => {
    try {
      const messageData = insertContactMessageSchema.parse(req.body);
      const message = await storage.createContactMessage(messageData);
      res.status(201).json({ message: "Message sent successfully" });
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
