'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { useLocation } from 'wouter';

interface NavbarProps {}

export function Navbar({}: NavbarProps) {
  const [scrolled, setScrolled] = useState(false);
  const [, setLocation] = useLocation(); // navigation function

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navItems = [
    { label: 'Home', href: '#home' },
    { label: 'Features', href: '#features' },
    { label: 'Technology', href: '#technology' },
    { label: 'Impact', href: '#impact' },
    { label: 'Contact', href: '#contact' },
  ];

  return (
    <motion.nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled ? 'glass-strong' : 'bg-transparent'
      }`}
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.8 }}
    >
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <motion.div
            className="flex items-center space-x-2"
            whileHover={{ scale: 1.05 }}
            transition={{ type: 'spring', stiffness: 400, damping: 10 }}
          >
            <div className="w-8 h-8 rounded-lg gradient-green-blue flex items-center justify-center">
              <span className="text-white font-bold text-lg">D</span>
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-[#00FF88] to-[#00FFFF] bg-clip-text text-transparent">
  KisanBuddy
</span>


          </motion.div>

          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item, index) => (
              <motion.a
                key={item.label}
                href={item.href}
                className="relative group text-white/80 hover:text-[var(--drone-green)] transition-colors duration-300"
                whileHover={{ y: -2 }}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                {item.label}
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-teal)] group-hover:w-full transition-all duration-300"></span>
              </motion.a>
            ))}
          </div>

          <motion.div
            className="flex items-center space-x-4"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
          >
            <Button
              variant="ghost"
              className="text-white/80 hover:text-[var(--drone-green)] border border-white/20 hover:border-[var(--drone-green)] transition-all duration-300"
              onClick={() => setLocation('/login')}
            >
              Login
            </Button>
            <Button
              className="bg-gradient-to-r from-[var(--drone-green)] to-[var(--drone-blue)] text-black font-medium px-6 py-2 rounded-lg hover:shadow-lg hover:shadow-[var(--drone-green)]/25 transition-all duration-300 hover:scale-105"
              style={{
                boxShadow: '0 0 20px rgba(0, 255, 136, 0.3)',
              }}
              onClick={() => setLocation('/signup')}
            >
              Register Now
            </Button>
          </motion.div>
        </div>
      </div>
    </motion.nav>
  );
}

