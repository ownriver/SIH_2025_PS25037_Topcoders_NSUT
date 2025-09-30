'use client';

import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import { Card } from '@/components/ui/card';
import { TrendingUp, Leaf, Droplets, Users } from 'lucide-react';
import { ImageWithFallback } from '@/components/ui/figma/ImageWithFallback';

const impacts = [
  {
    icon: TrendingUp,
    title: 'Increased Yield',
    value: '35%',
    description: 'Average crop yield improvement through precision fertilizer application and optimal nutrient distribution.',
    color: 'var(--drone-green)',
  },
  {
    icon: Leaf,
    title: 'Reduced Waste',
    value: '40%',
    description: 'Fertilizer waste reduction by targeting only areas that need nutrients, minimizing environmental impact.',
    color: 'var(--drone-teal)',
  },
  {
    icon: Droplets,
    title: 'Water Conservation',
    value: '25%',
    description: 'Water usage reduction through improved soil health and targeted nutrient application.',
    color: 'var(--drone-blue)',
  },
  {
    icon: Users,
    title: 'Farmer Success',
    value: '1000+',
    description: 'Farmers worldwide have adopted our technology to transform their agricultural practices.',
    color: 'var(--drone-green)',
  },
];

export function Impact() {
  const impactRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: impactRef,
    offset: ['start end', 'end start'],
  });

  const backgroundY = useTransform(scrollYProgress, [0, 1], ['-20%', '20%']);
  const overlayOpacity = useTransform(scrollYProgress, [0, 0.5, 1], [0.8, 0.5, 0.8]);

  return (
    <section
      ref={impactRef}
      id="impact"
      className="relative min-h-screen py-20 px-6 overflow-hidden"
    >
      {/* Parallax Background */}
      <motion.div
        className="absolute inset-0 z-0"
        style={{ y: backgroundY }}
      >
        <ImageWithFallback
          src="https://images.unsplash.com/photo-1744726010540-bf318d4a691f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzdXN0YWluYWJsZSUyMGZhcm1pbmclMjBoZWFsdGh5JTIwY3JvcHN8ZW58MXx8fHwxNzU4NDA2NzkwfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
          alt="Healthy sustainable crops"
          className="w-full h-full object-cover"
        />
      </motion.div>

      {/* Dynamic Overlay */}
      <motion.div
        className="absolute inset-0 z-5 bg-gradient-to-br from-[var(--drone-dark)]/90 via-[var(--drone-dark-secondary)]/80 to-[var(--drone-dark)]/90"
        style={{ opacity: overlayOpacity }}
      />

      {/* Animated Elements */}
      <div className="absolute inset-0 z-5">
        {[...Array(15)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute"
            initial={{
              x: Math.random() * window.innerWidth,
              y: window.innerHeight + 100,
              opacity: 0,
            }}
            animate={{
              y: -100,
              opacity: [0, 0.8, 0],
            }}
            transition={{
              duration: 8 + Math.random() * 4,
              repeat: Infinity,
              delay: Math.random() * 5,
              ease: 'linear',
            }}
          >
            <Leaf className="w-6 h-6 text-[var(--drone-green)]/30" />
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
              Measurable Impact
            </span>
          </h2>
          <p className="text-xl text-white/70 max-w-3xl mx-auto">
            See the real-world results that our Smart Fertilizer Drone technology delivers for farmers around the globe.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          {impacts.map((impact, index) => {
            const Icon = impact.icon;
            return (
              <motion.div
                key={impact.title}
                initial={{ opacity: 0, y: 50, scale: 0.9 }}
                whileInView={{ opacity: 1, y: 0, scale: 1 }}
                whileHover={{ 
                  y: -10,
                  scale: 1.05,
                }}
                transition={{ 
                  duration: 0.8, 
                  delay: index * 0.1,
                  hover: { duration: 0.3 }
                }}
                viewport={{ once: true }}
              >
                <Card className="glass-strong p-8 text-center h-full group hover:bg-white/10 transition-all duration-500">
                  <motion.div
                    className="w-20 h-20 mx-auto mb-6 rounded-full flex items-center justify-center relative"
                    style={{
                      background: `linear-gradient(135deg, ${impact.color}, ${impact.color}80)`,
                      boxShadow: `0 0 30px ${impact.color}40`,
                    }}
                    whileHover={{
                      boxShadow: `0 0 40px ${impact.color}60`,
                      scale: 1.1,
                    }}
                  >
                    <Icon className="w-10 h-10 text-white" />
                    <motion.div
                      className="absolute inset-0 rounded-full"
                      style={{
                        background: `radial-gradient(circle, ${impact.color}20, transparent 70%)`,
                      }}
                      animate={{
                        scale: [1, 1.2, 1],
                        opacity: [0.3, 0.6, 0.3],
                      }}
                      transition={{
                        duration: 3,
                        repeat: Infinity,
                      }}
                    />
                  </motion.div>

                  <motion.div
                    className="text-5xl font-bold mb-4"
                    style={{ color: impact.color }}
                    initial={{ scale: 0 }}
                    whileInView={{ scale: 1 }}
                    transition={{ 
                      duration: 0.6, 
                      delay: index * 0.1 + 0.3,
                      type: 'spring',
                      stiffness: 200,
                    }}
                    viewport={{ once: true }}
                  >
                    {impact.value}
                  </motion.div>

                  <h3 className="text-xl font-bold text-white mb-4 group-hover:text-[var(--drone-green)] transition-colors duration-300">
                    {impact.title}
                  </h3>

                  <p className="text-white/70 leading-relaxed group-hover:text-white/90 transition-colors duration-300">
                    {impact.description}
                  </p>
                </Card>
              </motion.div>
            );
          })}
        </div>

        {/* Success Story */}
        <motion.div
          className="glass p-8 md:p-12 rounded-2xl"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <h3 className="text-3xl md:text-4xl font-bold text-white mb-6">
                Transforming Agriculture
                <span className="text-[var(--drone-green)]"> Worldwide</span>
              </h3>
              <p className="text-lg text-white/80 mb-8 leading-relaxed">
                Our Smart Fertilizer Drone technology has revolutionized farming practices across diverse agricultural landscapes, from small family farms to large commercial operations. The precision and efficiency gains have been consistently remarkable.
              </p>
              
              <div className="grid grid-cols-2 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-[var(--drone-green)] mb-2">50+</div>
                  <div className="text-white/70">Countries</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-[var(--drone-teal)] mb-2">2M+</div>
                  <div className="text-white/70">Acres Covered</div>
                </div>
              </div>
            </div>

            <motion.div
              className="relative"
              whileHover={{ scale: 1.05 }}
              transition={{ duration: 0.3 }}
            >
              <ImageWithFallback
                src="https://media.istockphoto.com/id/617863246/photo/green-wheat-field.jpg?s=612x612&w=0&k=20&c=GAsectUgEou3pyByWrCyJcJ5TNdKGG0k9XBhqlMEoOE="
                alt="Green healthy crops"
                className="w-full h-80 object-cover rounded-xl"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent rounded-xl"></div>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}