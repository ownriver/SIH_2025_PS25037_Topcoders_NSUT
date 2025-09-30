'use client';

import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Brain, Radar, Cpu, Satellite, Camera, Gauge } from 'lucide-react';
import { ImageWithFallback } from '@/components/ui/figma/ImageWithFallback';

const technologies = [
  {
    icon: Brain,
    title: 'AI & Machine Learning',
    description: 'Advanced algorithms analyze crop patterns, soil conditions, and weather data to optimize fertilizer application in real-time.',
    features: ['Pattern Recognition', 'Predictive Analytics', 'Adaptive Learning'],
  },
  {
    icon: Radar,
    title: 'Precision Sensors',
    description: 'Multi-spectral sensors and LiDAR technology provide detailed field mapping and crop health monitoring.',
    features: ['Multispectral Imaging', 'LiDAR Mapping', 'Soil Analysis'],
  },
  {
    icon: Satellite,
    title: 'GPS Navigation',
    description: 'Centimeter-level accuracy ensures precise flight paths and fertilizer placement with RTK-GPS technology.',
    features: ['RTK-GPS', 'Autonomous Flight', 'Waypoint Navigation'],
  },
  {
    icon: Camera,
    title: 'Computer Vision',
    description: 'Real-time image processing identifies crop stress, disease patterns, and optimal application zones.',
    features: ['Disease Detection', 'Growth Monitoring', 'Quality Assessment'],
  },
];

export function Technology() {
  const techRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: techRef,
    offset: ['start end', 'end start'],
  });

  const backgroundY = useTransform(scrollYProgress, [0, 1], ['0%', '50%']);

  return (
    <section
      ref={techRef}
      id="technology"
      className="relative min-h-screen py-20 px-6 overflow-hidden"
    >
      {/* Animated Background */}
      <motion.div
        className="absolute inset-0 z-0"
        style={{ y: backgroundY }}
      >
        <ImageWithFallback
          src="https://images.unsplash.com/photo-1738598665806-7ecc32c3594c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwcmVjaXNpb24lMjBhZ3JpY3VsdHVyZSUyMHRlY2hub2xvZ3klMjBzZW5zb3JzfGVufDF8fHx8MTc1ODQwNjc4OHww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
          alt="Precision agriculture technology"
          className="w-full h-full object-cover opacity-20"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-[var(--drone-dark)] via-[var(--drone-dark-secondary)]/90 to-[var(--drone-dark)]"></div>
      </motion.div>

      {/* Floating Tech Elements */}
      <div className="absolute inset-0 z-5">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute"
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
              opacity: 0,
            }}
            animate={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
              opacity: [0, 0.6, 0],
            }}
            transition={{
              duration: 10 + Math.random() * 20,
              repeat: Infinity,
              ease: 'linear',
            }}
          >
            <div className="w-1 h-1 bg-[var(--drone-green)] rounded-full"></div>
          </motion.div>
        ))}
      </div>

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
              Advanced Technology
            </span>
          </h2>
          <p className="text-xl text-white/70 max-w-3xl mx-auto">
            Powered by cutting-edge AI, sensors, and automation systems designed for the future of precision agriculture.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
          {technologies.map((tech, index) => {
            const Icon = tech.icon;
            return (
              <motion.div
                key={tech.title}
                initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.8, delay: index * 0.2 }}
                viewport={{ once: true }}
              >
                <Card className="glass p-8 h-full group hover:glass-strong transition-all duration-500">
                  <div className="flex items-start space-x-6">
                    <motion.div
                      className="flex-shrink-0"
                      whileHover={{ rotate: 360 }}
                      transition={{ duration: 0.6 }}
                    >
                      <div
                        className="w-16 h-16 rounded-xl flex items-center justify-center"
                        style={{
                          background: `linear-gradient(135deg, var(--drone-green), var(--drone-teal))`,
                          boxShadow: '0 0 20px rgba(0, 255, 136, 0.3)',
                        }}
                      >
                        <Icon className="w-8 h-8 text-white" />
                      </div>
                    </motion.div>

                    <div className="flex-1">
                      <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-[var(--drone-green)] transition-colors duration-300">
                        {tech.title}
                      </h3>
                      <p className="text-white/70 mb-6 leading-relaxed">
                        {tech.description}
                      </p>
                      <div className="space-y-2">
                        {tech.features.map((feature, featureIndex) => (
                          <motion.div
                            key={feature}
                            className="flex items-center space-x-3"
                            initial={{ opacity: 0, x: -20 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.1 * featureIndex }}
                            viewport={{ once: true }}
                          >
                            <div className="w-2 h-2 bg-[var(--drone-green)] rounded-full"></div>
                            <span className="text-white/80">{feature}</span>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </div>
                </Card>
              </motion.div>
            );
          })}
        </div>

        {/* Technology Stats */}
        <motion.div
          className="grid grid-cols-2 md:grid-cols-4 gap-6"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          viewport={{ once: true }}
        >
          {[
            { label: 'Precision Accuracy', value: '99.8%', icon: Gauge },
            { label: 'Coverage Area', value: '500+', unit: 'acres/day', icon: Radar },
            { label: 'Efficiency Gain', value: '85%', icon: Cpu },
            { label: 'Data Points', value: '10M+', unit: 'per flight', icon: Brain },
          ].map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={stat.label}
                className="text-center glass p-6 rounded-xl hover:glass-strong transition-all duration-300"
                whileHover={{ y: -5, scale: 1.05 }}
                transition={{ delay: index * 0.1 }}
              >
                <Icon className="w-8 h-8 text-[var(--drone-green)] mx-auto mb-4" />
                <div className="text-3xl font-bold text-white mb-2">
                  {stat.value}
                  {stat.unit && <span className="text-lg text-white/60 ml-1">{stat.unit}</span>}
                </div>
                <div className="text-white/70">{stat.label}</div>
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    </section>
  );
}