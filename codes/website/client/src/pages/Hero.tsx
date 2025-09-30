'use client';

import { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Zap } from 'lucide-react';
import { ImageWithFallback } from '@/components/ui/figma/ImageWithFallback';

export function Hero() {
  const heroRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  });

  const y = useTransform(scrollYProgress, [0, 1], ['0%', '50%']);
  const opacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 1], [1, 1.1]);

  const droneX = useTransform(scrollYProgress, [0, 1], ['10%', '80%']);
  const droneY = useTransform(scrollYProgress, [0, 1], ['-10%', '-50%']);
  const droneRotate = useTransform(scrollYProgress, [0, 1], [0, 15]);

  return (
    <section
      ref={heroRef}
      id="home"
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      {/* Animated Background */}
      <motion.div className="absolute inset-0 z-0" style={{ y, scale }}>
        <div className="absolute inset-0 bg-gradient-to-br from-[var(--drone-dark)] via-[var(--drone-dark-secondary)] to-black"></div>
        <ImageWithFallback
          src="https://images.unsplash.com/photo-1559506709-e3d879c60305?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkcm9uZSUyMGFncmljdWx0dXJlJTIwZmFybWxhbmQlMjBhZXJpYWx8ZW58MXx8fHwxNzU4NDA2NzgzfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
          alt="Aerial view of farmland"
          className="w-full h-full object-cover opacity-30"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-[var(--drone-dark)] via-transparent to-[var(--drone-dark)]/50"></div>
      </motion.div>

      {/* Animated Particles */}
      <div className="absolute inset-0 z-10">
        {[...Array(50)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-[var(--drone-green)] rounded-full opacity-60"
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
            }}
            animate={{
              y: [0, -20, 0],
              opacity: [0.3, 1, 0.3],
            }}
            transition={{
              duration: 2 + Math.random() * 3,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      {/* Flying Drone */}
      <motion.div
        className="absolute top-1/4 z-20 pointer-events-none"
        style={{
          x: droneX,
          y: droneY,
          rotate: droneRotate,
        }}
        initial={{ x: '-100%', opacity: 0 }}
        animate={{ x: '10%', opacity: 1 }}
        transition={{ duration: 2, ease: 'easeOut' }}
      >
        <motion.div
          className="relative"
          animate={{ y: [0, -10, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        >
          <div className="w-20 h-20 md:w-32 md:h-32">
            <svg viewBox="0 0 100 100" className="w-full h-full">
              {/* Drone body */}
              <ellipse cx="50" cy="50" rx="15" ry="8" fill="var(--drone-green)" />

              {/* Propellers */}
              <motion.g
                animate={{ rotate: 360 }}
                transition={{ duration: 0.1, repeat: Infinity, ease: 'linear' }}
                style={{ transformOrigin: '25px 35px' }}
              >
                <circle cx="25" cy="35" r="8" fill="rgba(255,255,255,0.3)" />
              </motion.g>
              <motion.g
                animate={{ rotate: -360 }}
                transition={{ duration: 0.1, repeat: Infinity, ease: 'linear' }}
                style={{ transformOrigin: '75px 35px' }}
              >
                <circle cx="75" cy="35" r="8" fill="rgba(255,255,255,0.3)" />
              </motion.g>
              <motion.g
                animate={{ rotate: 360 }}
                transition={{ duration: 0.1, repeat: Infinity, ease: 'linear' }}
                style={{ transformOrigin: '25px 65px' }}
              >
                <circle cx="25" cy="65" r="8" fill="rgba(255,255,255,0.3)" />
              </motion.g>
              <motion.g
                animate={{ rotate: -360 }}
                transition={{ duration: 0.1, repeat: Infinity, ease: 'linear' }}
                style={{ transformOrigin: '75px 65px' }}
              >
                <circle cx="75" cy="65" r="8" fill="rgba(255,255,255,0.3)" />
              </motion.g>

              {/* Arms */}
              <line x1="35" y1="40" x2="25" y2="35" stroke="var(--drone-teal)" strokeWidth="2" />
              <line x1="65" y1="40" x2="75" y2="35" stroke="var(--drone-teal)" strokeWidth="2" />
              <line x1="35" y1="60" x2="25" y2="65" stroke="var(--drone-teal)" strokeWidth="2" />
              <line x1="65" y1="60" x2="75" y2="65" stroke="var(--drone-teal)" strokeWidth="2" />
            </svg>

            {/* Spray effect */}
            <motion.div
              className="absolute -bottom-4 left-1/2 transform -translate-x-1/2"
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 1, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <div className="w-2 h-8 bg-gradient-to-b from-[var(--drone-green)]/50 to-transparent"></div>
            </motion.div>
          </div>
        </motion.div>
      </motion.div>

      {/* Hero Content */}
      <motion.div
        className="relative z-30 text-center max-w-5xl mx-auto px-6"
        style={{ opacity }}
      >
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.5 }}
        >
          {/* Professional Hero Title */}
          <motion.h1
            className="text-5xl md:text-6xl lg:text-7xl font-extrabold mb-6 leading-tight relative text-transparent bg-clip-text"
            style={{
              backgroundImage: 'linear-gradient(90deg, #16A34A, #22D3EE)',
              textShadow: '0 2px 8px rgba(0,0,0,0.5)',
            }}
          >
            Smart Pesticide Drone
          </motion.h1>

          <motion.h2
            className="text-2xl md:text-3xl lg:text-4xl text-white/80 mb-8 font-medium"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.7 }}
          >
            for Precision Agriculture
          </motion.h2>
        </motion.div>

        <motion.p
          className="text-lg md:text-xl text-white/70 mb-12 max-w-3xl mx-auto leading-relaxed"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.9 }}
        >
          Revolutionize your farming with AI-powered drone technology that
          delivers precise fertilizer application, reduces waste, and maximizes
          crop yield.
        </motion.p>

        {/* Buttons */}
        <motion.div
          className="flex flex-col sm:flex-row gap-6 justify-center"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1.1 }}
        >
          {/* Learn More */}
          <Button
            size="lg"
            onClick={() => {
              const el = document.getElementById('features');
              el?.scrollIntoView({ behavior: 'smooth' });
            }}
            className="bg-gradient-to-r from-green-500 to-teal-400 text-white font-semibold px-8 py-4 text-lg rounded-xl shadow-lg hover:shadow-green-500/30 hover:scale-105 transition-all duration-300"
          >
            <Zap className="mr-2" />
            Learn More
          </Button>

          {/* Watch Demo */}
          <Button
            variant="outline"
            size="lg"
            onClick={() =>
              window.open('https://www.youtube.com/watch?v=WZO1AtQHKKk', '_blank')
            }
            className="border-2 border-green-400 text-green-400 hover:bg-green-400 hover:text-black font-semibold px-8 py-4 text-lg rounded-xl transition-all duration-300"
          >
            Watch Demo
          </Button>
        </motion.div>
      </motion.div>
    </section>
  );
}
