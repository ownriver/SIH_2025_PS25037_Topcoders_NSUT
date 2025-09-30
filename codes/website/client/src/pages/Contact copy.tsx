'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';
import { Mail, Phone, MapPin, Send, CheckCircle } from 'lucide-react';

export function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    company: '',
    message: '',
  });
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitted(true);
    // Simulate form submission
    setTimeout(() => {
      setIsSubmitted(false);
      setFormData({ name: '', email: '', company: '', message: '' });
    }, 3000);
  };

  const contactInfo = [
    {
      icon: Mail,
      title: 'Email Us',
      value: 'contact@KisanBuddy.com',
      description: 'Get in touch with our team',
    },
    {
      icon: Phone,
      title: 'Call Us',
      value: '+1 (555) 123-4567',
      description: '24/7 support available',
    },
    {
      icon: MapPin,
      title: 'Visit Us',
      value: 'San Francisco, CA',
      description: 'Innovation headquarters',
    },
  ];

  return (
    <section id="contact" className="relative min-h-screen py-20 px-6">
      {/* Background */}
      <div className="absolute inset-0 gradient-dark"></div>
      <div className="absolute inset-0 bg-gradient-to-br from-[var(--drone-green)]/5 via-transparent to-[var(--drone-blue)]/5"></div>

      <div className="relative z-10 max-w-7xl mx-auto">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-4xl md:text-6xl font-bold mb-6">
            <span className="bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-blue)] bg-clip-text text-transparent">
              Get Started Today
            </span>
          </h2>
          <p className="text-xl text-white/70 max-w-3xl mx-auto">
            Ready to revolutionize your farming operations? Contact our team to learn more about our Smart Fertilizer Drone solutions.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <Card className="glass-strong p-8">
              <h3 className="text-2xl font-bold text-white mb-8">Send us a message</h3>
              
              {isSubmitted ? (
                <motion.div
                  className="text-center py-12"
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <CheckCircle className="w-16 h-16 text-[var(--drone-green)] mx-auto mb-4" />
                  <h4 className="text-xl font-bold text-white mb-2">Message Sent!</h4>
                  <p className="text-white/70">We'll get back to you within 24 hours.</p>
                </motion.div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <motion.div
                      className="relative"
                      whileHover={{ scale: 1.02 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Input
                        type="text"
                        placeholder="Your Name"
                        value={formData.name}
                        onChange={(e) => handleInputChange('name', e.target.value)}
                        onFocus={() => setFocusedField('name')}
                        onBlur={() => setFocusedField(null)}
                        className={`glass border-white/20 text-white placeholder-white/50 transition-all duration-300 ${
                          focusedField === 'name' ? 'border-[var(--drone-green)] shadow-lg shadow-[var(--drone-green)]/20' : ''
                        }`}
                        required
                      />
                    </motion.div>

                    <motion.div
                      className="relative"
                      whileHover={{ scale: 1.02 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Input
                        type="email"
                        placeholder="Email Address"
                        value={formData.email}
                        onChange={(e) => handleInputChange('email', e.target.value)}
                        onFocus={() => setFocusedField('email')}
                        onBlur={() => setFocusedField(null)}
                        className={`glass border-white/20 text-white placeholder-white/50 transition-all duration-300 ${
                          focusedField === 'email' ? 'border-[var(--drone-green)] shadow-lg shadow-[var(--drone-green)]/20' : ''
                        }`}
                        required
                      />
                    </motion.div>
                  </div>

                  <motion.div
                    className="relative"
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Input
                      type="text"
                      placeholder="Company Name (Optional)"
                      value={formData.company}
                      onChange={(e) => handleInputChange('company', e.target.value)}
                      onFocus={() => setFocusedField('company')}
                      onBlur={() => setFocusedField(null)}
                      className={`glass border-white/20 text-white placeholder-white/50 transition-all duration-300 ${
                        focusedField === 'company' ? 'border-[var(--drone-green)] shadow-lg shadow-[var(--drone-green)]/20' : ''
                      }`}
                    />
                  </motion.div>

                  <motion.div
                    className="relative"
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Textarea
                      placeholder="Tell us about your farming operation and how we can help..."
                      rows={6}
                      value={formData.message}
                      onChange={(e) => handleInputChange('message', e.target.value)}
                      onFocus={() => setFocusedField('message')}
                      onBlur={() => setFocusedField(null)}
                      className={`glass border-white/20 text-white placeholder-white/50 transition-all duration-300 resize-none ${
                        focusedField === 'message' ? 'border-[var(--drone-green)] shadow-lg shadow-[var(--drone-green)]/20' : ''
                      }`}
                      required
                    />
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Button
                      type="submit"
                      className="w-full bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-teal)] text-black font-medium py-3 rounded-xl hover:shadow-lg hover:shadow-[var(--drone-green)]/25 transition-all duration-300"
                    >
                      <Send className="w-5 h-5 mr-2" />
                      Send Message
                    </Button>
                  </motion.div>
                </form>
              )}
            </Card>
          </motion.div>

          {/* Contact Info */}
          <motion.div
            className="space-y-8"
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            {contactInfo.map((info, index) => {
              const Icon = info.icon;
              return (
                <motion.div
                  key={info.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  whileHover={{ scale: 1.05, x: 10 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card className="glass p-6 hover:glass-strong transition-all duration-300">
                    <div className="flex items-center space-x-4">
                      <div
                        className="w-14 h-14 rounded-xl flex items-center justify-center"
                        style={{
                          background: 'linear-gradient(135deg, var(--drone-green), var(--drone-teal))',
                          boxShadow: '0 0 20px rgba(0, 255, 136, 0.3)',
                        }}
                      >
                        <Icon className="w-7 h-7 text-white" />
                      </div>
                      <div>
                        <h4 className="text-lg font-bold text-white mb-1">{info.title}</h4>
                        <p className="text-[var(--drone-green)] font-medium mb-1">{info.value}</p>
                        <p className="text-white/60 text-sm">{info.description}</p>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              );
            })}

            {/* CTA */}
            <motion.div
              className="glass p-8 rounded-xl text-center"
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              viewport={{ once: true }}
            >
              <h4 className="text-xl font-bold text-white mb-4">
                Ready to Transform Your Farm?
              </h4>
              <p className="text-white/70 mb-6">
                Schedule a personalized demo and see our technology in action.
              </p>
              <Button
                className="bg-gradient-to-r from-[var(--drone-blue)] to-[var(--drone-teal)] text-white font-medium px-8 py-3 rounded-xl hover:shadow-lg hover:shadow-[var(--drone-blue)]/25 transition-all duration-300"
              >
                Schedule Demo
              </Button>
            </motion.div>
          </motion.div>
        </div>

        {/* Footer */}
        <motion.footer
          className="mt-20 pt-12 border-t border-white/10 text-center"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="flex items-center justify-center space-x-2 mb-4">
            <div className="w-8 h-8 rounded-lg gradient-green-blue flex items-center justify-center">
              <span className="text-white font-bold">D</span>
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-teal)] bg-clip-text text-transparent">
              KisanBuddy
            </span>
          </div>
          <p className="text-white/60 mb-6">
            Revolutionizing agriculture through intelligent automation and precision technology.
          </p>
          <div className="flex justify-center space-x-8 text-white/40">
            <span>&copy; 2024 KisanBuddy. All rights reserved.</span>
            <span>Privacy Policy</span>
            <span>Terms of Service</span>
          </div>
        </motion.footer>
      </div>
    </section>
  );
}