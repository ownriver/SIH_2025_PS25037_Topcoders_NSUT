import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, real, timestamp, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  email: text("email").notNull().unique(),
  password: text("password").notNull(),
  fullName: text("full_name").notNull(),
  role: text("role").notNull().default("farmer"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const crops = pgTable("crops", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  name: text("name").notNull(),
  type: text("type").notNull(), // rice, wheat, cotton, corn
  plantedDate: timestamp("planted_date").notNull(),
  expectedHarvestDate: timestamp("expected_harvest_date").notNull(),
  growthStage: text("growth_stage").notNull().default("seedling"),
  area: real("area").notNull(), // in hectares
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
});

export const fields = pgTable("fields", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  cropId: varchar("crop_id").references(() => crops.id),
  name: text("name").notNull(),
  latitude: real("latitude").notNull(),
  longitude: real("longitude").notNull(),
  area: real("area").notNull(), // in hectares
  boundaries: text("boundaries"), // JSON string of polygon coordinates
  createdAt: timestamp("created_at").defaultNow(),
});

export const droneConnections = pgTable("drone_connections", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  droneName: text("drone_name").notNull(),
  connectionType: text("connection_type").notNull(), // wifi, bluetooth
  status: text("status").notNull().default("connected"), // connected, disconnected, scanning
  batteryLevel: integer("battery_level").default(100),
  lastSeen: timestamp("last_seen").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const plantHealthRecords = pgTable("plant_health_records", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  fieldId: varchar("field_id").notNull().references(() => fields.id),
  droneId: varchar("drone_id").references(() => droneConnections.id),
  healthScore: integer("health_score").notNull(), // 0-100
  infectionRate: real("infection_rate").notNull(), // percentage
  infectionType: text("infection_type"), // aphid, fungal, bacterial, etc.
  severity: text("severity").notNull(), // low, medium, high
  latitude: real("latitude"),
  longitude: real("longitude"),
  detectionConfidence: integer("detection_confidence").notNull(), // 0-100
  recordedAt: timestamp("recorded_at").defaultNow(),
});

export const pesticideApplications = pgTable("pesticide_applications", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  fieldId: varchar("field_id").notNull().references(() => fields.id),
  healthRecordId: varchar("health_record_id").references(() => plantHealthRecords.id),
  pesticideType: text("pesticide_type").notNull(),
  volumePerHectare: real("volume_per_hectare").notNull(), // liters
  totalVolume: real("total_volume").notNull(), // liters
  applicationMethod: text("application_method").notNull().default("drone"),
  status: text("status").notNull().default("recommended"), // recommended, scheduled, applied, completed
  recommendedBy: text("recommended_by").default("ai_system"),
  confidence: integer("confidence").notNull(), // 0-100
  scheduledFor: timestamp("scheduled_for"),
  appliedAt: timestamp("applied_at"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const contactMessages = pgTable("contact_messages", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  email: text("email").notNull(),
  message: text("message").notNull(),
  status: text("status").default("new"),
  createdAt: timestamp("created_at").defaultNow(),
});

// Insert schemas
export const insertUserSchema = createInsertSchema(users).omit({
  id: true,
  createdAt: true,
});

export const insertCropSchema = createInsertSchema(crops).omit({
  id: true,
  createdAt: true,
});

export const insertFieldSchema = createInsertSchema(fields).omit({
  id: true,
  createdAt: true,
});

export const insertDroneConnectionSchema = createInsertSchema(droneConnections).omit({
  id: true,
  createdAt: true,
  lastSeen: true,
});

export const insertPlantHealthRecordSchema = createInsertSchema(plantHealthRecords).omit({
  id: true,
  recordedAt: true,
});

export const insertPesticideApplicationSchema = createInsertSchema(pesticideApplications).omit({
  id: true,
  createdAt: true,
});

export const insertContactMessageSchema = createInsertSchema(contactMessages).omit({
  id: true,
  createdAt: true,
});

// Login schema
export const loginSchema = z.object({
  username: z.string().min(1, "Username is required"),
  password: z.string().min(1, "Password is required"),
});

// Types
export type User = typeof users.$inferSelect;
export type InsertUser = z.infer<typeof insertUserSchema>;
export type Crop = typeof crops.$inferSelect;
export type InsertCrop = z.infer<typeof insertCropSchema>;
export type Field = typeof fields.$inferSelect;
export type InsertField = z.infer<typeof insertFieldSchema>;
export type DroneConnection = typeof droneConnections.$inferSelect;
export type InsertDroneConnection = z.infer<typeof insertDroneConnectionSchema>;
export type PlantHealthRecord = typeof plantHealthRecords.$inferSelect;
export type InsertPlantHealthRecord = z.infer<typeof insertPlantHealthRecordSchema>;
export type PesticideApplication = typeof pesticideApplications.$inferSelect;
export type InsertPesticideApplication = z.infer<typeof insertPesticideApplicationSchema>;
export type ContactMessage = typeof contactMessages.$inferSelect;
export type InsertContactMessage = z.infer<typeof insertContactMessageSchema>;
export type LoginCredentials = z.infer<typeof loginSchema>;
