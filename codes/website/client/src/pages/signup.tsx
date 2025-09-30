'use client';

import { useState } from "react";
import { useLocation } from "wouter";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { User, AtSign, Lock } from "lucide-react";

export default function SignupPage() {
  const { signup } = useAuth();
  const [, setLocation] = useLocation();

  const [formData, setFormData] = useState({
    username: "",
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: ""
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      await signup({
        username: formData.username,
        email: formData.email,
        password: formData.password,
        fullName: `${formData.firstName} ${formData.lastName}`,
      });
      setLocation("/dashboard");
    } catch (err: any) {
      setError(err?.message || "Signup failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0B0C10] px-4">
      <Card className="w-full max-w-md p-10 rounded-3xl glass-strong shadow-xl bg-gradient-to-b from-[#111315] to-[#1A1C20] border border-white/10">
        {/* Logo */}
        <div className="flex justify-center mb-6">
          <div className="w-20 h-20 bg-gradient-to-tr from-green-400 to-teal-400 rounded-full flex items-center justify-center text-black text-2xl font-bold shadow-lg">
            DN
          </div>
        </div>

        <h2 className="text-3xl font-bold mb-6 text-center text-green-400">Create Account</h2>

        {error && <p className="text-red-500 mb-4 text-center">{error}</p>}

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Username */}
          <div className="relative">
            <User className="absolute top-1/2 left-3 -translate-y-1/2 text-white/50" size={20} />
            <Input
              type="text"
              placeholder="Username"
              value={formData.username}
              onChange={(e) => handleChange("username", e.target.value)}
              required
              className="pl-10 text-white bg-[#1A1C20] border border-white/20 focus:ring-green-400 focus:border-green-400"
            />
          </div>

          {/* First & Last Name */}
          <div className="grid grid-cols-2 gap-4">
            <div className="relative">
              <User className="absolute top-1/2 left-3 -translate-y-1/2 text-white/50" size={20} />
              <Input
                type="text"
                placeholder="First Name"
                value={formData.firstName}
                onChange={(e) => handleChange("firstName", e.target.value)}
                required
                className="pl-10 text-white bg-[#1A1C20] border border-white/20 focus:ring-green-400 focus:border-green-400"
              />
            </div>
            <div className="relative">
              <User className="absolute top-1/2 left-3 -translate-y-1/2 text-white/50" size={20} />
              <Input
                type="text"
                placeholder="Last Name"
                value={formData.lastName}
                onChange={(e) => handleChange("lastName", e.target.value)}
                required
                className="pl-10 text-white bg-[#1A1C20] border border-white/20 focus:ring-green-400 focus:border-green-400"
              />
            </div>
          </div>

          {/* Email */}
          <div className="relative">
            <AtSign className="absolute top-1/2 left-3 -translate-y-1/2 text-white/50" size={20} />
            <Input
              type="email"
              placeholder="Email Address"
              value={formData.email}
              onChange={(e) => handleChange("email", e.target.value)}
              required
              className="pl-10 text-white bg-[#1A1C20] border border-white/20 focus:ring-green-400 focus:border-green-400"
            />
          </div>

          {/* Password */}
          <div className="relative">
            <Lock className="absolute top-1/2 left-3 -translate-y-1/2 text-white/50" size={20} />
            <Input
              type="password"
              placeholder="Password"
              value={formData.password}
              onChange={(e) => handleChange("password", e.target.value)}
              required
              className="pl-10 text-white bg-[#1A1C20] border border-white/20 focus:ring-green-400 focus:border-green-400"
            />
          </div>

          {/* Confirm Password */}
          <div className="relative">
            <Lock className="absolute top-1/2 left-3 -translate-y-1/2 text-white/50" size={20} />
            <Input
              type="password"
              placeholder="Confirm Password"
              value={formData.confirmPassword}
              onChange={(e) => handleChange("confirmPassword", e.target.value)}
              required
              className="pl-10 text-white bg-[#1A1C20] border border-white/20 focus:ring-green-400 focus:border-green-400"
            />
          </div>

          {/* Submit Button */}
          <Button
            type="submit"
            className="w-full py-3 bg-gradient-to-r from-green-400 to-teal-400 text-black font-semibold rounded-xl shadow-lg hover:scale-105 transition-transform duration-300"
            disabled={loading}
          >
            {loading ? "Signing up..." : "Create Account"}
          </Button>
        </form>

        <p className="mt-6 text-center text-sm text-white/60">
          Already have an account?{" "}
          <a href="/login" className="text-green-400 hover:underline">
            Login
          </a>
        </p>
      </Card>
    </div>
  );
}
