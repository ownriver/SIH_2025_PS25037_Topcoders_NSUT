'use client';

import { useState } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Eye, EyeOff, ArrowLeft, Zap, Shield, User } from "lucide-react";
import { useAuth } from "@/hooks/use-auth";
import { ImageWithFallback } from "@/components/ui/figma/ImageWithFallback";

interface LoginPageProps {
  onBack: () => void;
}

export function LoginPage({ onBack }: LoginPageProps) {
  const { login } = useAuth();
  const [, setLocation] = useLocation();

  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    company: '',
  });

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!isLogin && formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    if (isLogin) {
      setLoading(true);
      try {
        await login({ username: formData.email, password: formData.password });
        setLocation("/dashboard"); // redirect after login
      } catch (err: any) {
        setError(err.message || "Login failed. Please try again.");
      } finally {
        setLoading(false);
      }
    } else {
      // TODO: add signup logic
      console.log("Signup data:", formData);
      alert("Signup feature coming soon!");
    }
  };

  return (
    <div className="min-h-screen bg-[var(--drone-dark)] text-white overflow-hidden relative">
      {/* Animated Background */}
      <motion.div
        className="absolute inset-0 z-0"
        initial={{ scale: 1.1, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 1.5 }}
      >
        <ImageWithFallback
          src="https://images.unsplash.com/photo-1559506709-e3d879c60305?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=1080"
          alt="Drone agriculture background"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-[var(--drone-dark)]/95 via-[var(--drone-dark-secondary)]/90 to-[var(--drone-dark)]/95"></div>
      </motion.div>

      {/* Floating Particles */}
      <div className="absolute inset-0 z-5">
        {[...Array(30)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-[var(--drone-green)] rounded-full opacity-40"
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
            }}
            animate={{
              y: [0, -20, 0],
              x: [0, Math.random() * 50 - 25, 0],
              opacity: [0.2, 0.8, 0.2],
            }}
            transition={{
              duration: 3 + Math.random() * 4,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      {/* Back Button */}
      <motion.button
        className="absolute top-8 left-8 z-20 flex items-center space-x-2 text-white/70 hover:text-[var(--drone-green)] transition-colors duration-300"
        onClick={onBack}
      >
        <ArrowLeft className="w-5 h-5" />
        <span>Back to Home</span>
      </motion.button>

      {/* Login Container */}
      <div className="relative z-10 min-h-screen flex items-center justify-center px-6">
        <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left Side - Branding */}
          <motion.div className="text-center lg:text-left">
            <motion.div className="flex items-center justify-center lg:justify-start space-x-3 mb-8">
              <div className="w-12 h-12 rounded-xl gradient-green-blue flex items-center justify-center">
                <span className="text-white font-bold text-xl">D</span>
              </div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-teal)] bg-clip-text text-transparent">
                KisanBuddy
              </h1>
            </motion.div>

            <h2 className="text-4xl lg:text-6xl font-bold mb-6">
              Welcome to the
              <span className="block bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-blue)] bg-clip-text text-transparent">
                Future of Farming
              </span>
            </h2>

            <p className="text-xl text-white/70 mb-8 max-w-lg">
              Join thousands of farmers revolutionizing agriculture with our AI-powered drones.
            </p>
          </motion.div>

          {/* Right Side - Login Form */}
          <motion.div>
            <Card className="glass-strong p-8 lg:p-12 max-w-md mx-auto">
              {error && <p className="text-red-400 mb-4">{error}</p>}

              {/* Toggle Buttons */}
              <div className="flex mb-8">
                <Button
                  variant={isLogin ? 'default' : 'ghost'}
                  className={`flex-1 mr-2 ${isLogin ? 'bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-teal)] text-black' : 'text-white/70 hover:text-white'}`}
                  onClick={() => setIsLogin(true)}
                >
                  Sign In
                </Button>
                <Button
                  variant={!isLogin ? 'default' : 'ghost'}
                  className={`flex-1 ml-2 ${!isLogin ? 'bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-teal)] text-black' : 'text-white/70 hover:text-white'}`}
                  onClick={() => setIsLogin(false)}
                >
                  Sign Up
                </Button>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                <Input
                  type="email"
                  placeholder="Email Address"
                  value={formData.email}
                  onChange={(e) => handleInputChange('email', e.target.value)}
                  required
                />

                <div className="relative">
                  <Input
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Password"
                    value={formData.password}
                    onChange={(e) => handleInputChange('password', e.target.value)}
                    required
                  />
                  <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/60 hover:text-[var(--drone-green)]">
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>

                <Button type="submit" className="w-full py-3 bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-teal)] text-black font-medium rounded-xl hover:shadow-lg">
                  {loading ? "Logging in..." : isLogin ? "Sign In" : "Create Account"}
                </Button>
              </form>

              <div className="mt-8 text-center">
                <p className="text-white/60 text-sm">
                  {isLogin ? "Don't have an account? " : "Already have an account? "}
                  <button onClick={() => setIsLogin(!isLogin)} className="text-[var(--drone-green)] hover:text-[var(--drone-teal)] font-medium">
                    {isLogin ? 'Sign Up' : 'Sign In'}
                  </button>
                </p>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
