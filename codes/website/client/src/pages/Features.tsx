'use client';

import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Leaf, Target, DollarSign, Zap, Droplets, BarChart3 } from 'lucide-react';

const features = [
  {
    icon: Target,
    title: 'Precision Spraying',
    description: 'Advanced GPS and sensor technology ensure fertilizer is applied exactly where needed, minimizing waste and maximizing effectiveness.',
    color: 'var(--drone-green)',
  },
  {
    icon: Leaf,
    title: 'Reduced Environmental Impact',
    description: 'Minimize fertilizer runoff and reduce chemical usage by up to 40% while maintaining optimal crop nutrition.',
    color: 'var(--drone-teal)',
  },
  {
    icon: DollarSign,
    title: 'Cost Efficiency',
    description: 'Save up to 60% on labor costs and 30% on fertilizer expenses with automated, precise application technology.',
    color: 'var(--drone-blue)',
  },
  {
    icon: Zap,
    title: 'Real-time Monitoring',
    description: 'Monitor crop health, soil conditions, and application progress in real-time through our advanced dashboard.',
    color: 'var(--drone-green)',
  },
  {
    icon: Droplets,
    title: 'Smart Application',
    description: 'AI algorithms analyze soil moisture, nutrient levels, and crop stage to determine optimal fertilizer mix and timing.',
    color: 'var(--drone-teal)',
  },
  {
    icon: BarChart3,
    title: 'Data Analytics',
    description: 'Comprehensive analytics and reporting help optimize your farming strategy and track ROI over time.',
    color: 'var(--drone-blue)',
  },
];

export function Features() {
  const featuresRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: featuresRef,
    offset: ['start end', 'end start'],
  });

  const y = useTransform(scrollYProgress, [0, 1], [50, -50]);

  return (
    <section
      ref={featuresRef}
      id="features"
      className="relative min-h-screen py-20 px-6"
    >
      {/* Background */}
      <div className="absolute inset-0 gradient-dark"></div>
      <motion.div
        className="absolute inset-0 opacity-20"
        style={{ y }}
      >
        <div className="w-full h-full bg-gradient-to-br from-[var(--drone-green)]/10 via-transparent to-[var(--drone-blue)]/10"></div>
      </motion.div>

      <div className="relative z-10 max-w-7xl mx-auto">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-4xl md:text-6xl font-bold mb-6">
            <span
              className="bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-blue)] bg-clip-text text-transparent"
            >
              Revolutionary Features
            </span>
          </h2>
          <p className="text-xl text-white/70 max-w-3xl mx-auto">
            Experience the future of precision agriculture with our cutting-edge drone technology designed to optimize every aspect of fertilizer application.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 50, scale: 0.9 }}
                whileInView={{ opacity: 1, y: 0, scale: 1 }}
                whileHover={{ 
                  y: -10, 
                  scale: 1.05,
                  transition: { duration: 0.3 }
                }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="glass p-8 h-full hover:glass-strong transition-all duration-500 border-white/10 group">
                  <div className="relative">
                    <motion.div
                      className="w-16 h-16 rounded-xl flex items-center justify-center mb-6 relative overflow-hidden"
                      style={{
                        background: `linear-gradient(135deg, ${feature.color}, ${feature.color}80)`,
                        boxShadow: `0 0 20px ${feature.color}40`,
                      }}
                      whileHover={{
                        boxShadow: `0 0 30px ${feature.color}60`,
                        scale: 1.1,
                      }}
                      transition={{ duration: 0.3 }}
                    >
                      <Icon className="w-8 h-8 text-white relative z-10" />
                      <motion.div
                        className="absolute inset-0 opacity-20"
                        style={{
                          background: `radial-gradient(circle, ${feature.color}, transparent)`,
                        }}
                        animate={{
                          scale: [1, 1.2, 1],
                          opacity: [0.2, 0.4, 0.2],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                        }}
                      />
                    </motion.div>

                    <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-[var(--drone-green)] transition-colors duration-300">
                      {feature.title}
                    </h3>
                    
                    <p className="text-white/70 leading-relaxed group-hover:text-white/90 transition-colors duration-300">
                      {feature.description}
                    </p>

                    {/* Hover glow effect */}
                    <motion.div
                      className="absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-500"
                      style={{
                        background: `radial-gradient(circle at center, ${feature.color}10, transparent 70%)`,
                      }}
                    />
                  </div>
                </Card>
              </motion.div>
            );
          })}
        </div>

        {/* Bottom CTA */}
        <motion.div
          className="text-center mt-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          viewport={{ once: true }}
        >
          <p className="text-lg text-white/60 mb-6">
            Ready to transform your agricultural practices?
          </p>
          <motion.button
            className="px-8 py-4 bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-teal)] text-black font-medium rounded-xl hover:shadow-lg hover:shadow-[var(--drone-green)]/25 transition-all duration-300"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Explore All Features
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}