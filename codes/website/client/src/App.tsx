'use client';

import { Switch, Route } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useAuth } from "@/hooks/use-auth";

// New Figma UI pages
import LandingPage from "@/pages/landing";
import LoginPage from "@/pages/login";
import SignupPage from "@/pages/signup";

// Existing pages after login
import DashboardPage from "@/pages/dashboard";
import CropSelectionPage from "@/pages/crop-selection";
import DroneConnectionPage from "@/pages/drone-connection";
import FieldMapPage from "@/pages/field-map";
import PlantHealthPage from "@/pages/plant-health";
import PesticideControlPage from "@/pages/pesticide-control";
import ContactPage from "@/pages/contact";
import NotFound from "@/pages/not-found";

import { SidebarLayout } from "@/components/ui/sidebar-layout";

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LandingPage />;
  }

  return <SidebarLayout>{children}</SidebarLayout>;
}

function Router() {
  return (
    <Switch>
      {/* Public routes */}
      <Route path="/" component={LandingPage} />
      <Route path="/login" component={LoginPage} />
      <Route path="/signup" component={SignupPage} />

      {/* Protected routes */}
      <Route path="/dashboard">
        <ProtectedRoute>
          <DashboardPage />
        </ProtectedRoute>
      </Route>
      <Route path="/crop-selection">
        <ProtectedRoute>
          <CropSelectionPage />
        </ProtectedRoute>
      </Route>
      <Route path="/drone-connection">
        <ProtectedRoute>
          <DroneConnectionPage />
        </ProtectedRoute>
      </Route>
      <Route path="/field-map">
        <ProtectedRoute>
          <FieldMapPage />
        </ProtectedRoute>
      </Route>
      <Route path="/plant-health">
        <ProtectedRoute>
          <PlantHealthPage />
        </ProtectedRoute>
      </Route>
      <Route path="/pesticide-control">
        <ProtectedRoute>
          <PesticideControlPage />
        </ProtectedRoute>
      </Route>
      <Route path="/contact">
        <ProtectedRoute>
          <ContactPage />
        </ProtectedRoute>
      </Route>

      {/* Catch-all */}
      <Route component={NotFound} />
    </Switch>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Router />
      </TooltipProvider>
    </QueryClientProvider>
  );
}