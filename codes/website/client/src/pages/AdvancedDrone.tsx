'use client';

import { useEffect, useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import exampleImage from 'figma:asset/dc5cfa40ad391d62ad8246d77c776467ef324698.png';

export function AdvancedDrone() {
  const droneRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll();
  
  // Autonomous flight patterns - no mouse following
  const autonomousX = useTransform(scrollYProgress, 
    [0, 0.2, 0.4, 0.6, 0.8, 1], 
    ['15%', '85%', '25%', '75%', '35%', '20%']
  );
  const autonomousY = useTransform(scrollYProgress, 
    [0, 0.2, 0.4, 0.6, 0.8, 1], 
    ['20%', '35%', '65%', '25%', '70%', '40%']
  );
  
  const rotateZ = useTransform(scrollYProgress, [0, 1], [0, 360]);

  return (
    <motion.div
      ref={droneRef}
      className="fixed z-40 pointer-events-none"
      style={{
        x: autonomousX,
        y: autonomousY,
        rotateZ: rotateZ,
        left: '0%',
        top: '0%',
      }}
      initial={{ opacity: 0, scale: 0, y: -100 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 2, ease: 'easeOut' }}
    >
      <motion.div
        className="relative w-24 h-24 md:w-32 md:h-32"
        animate={{
          y: [0, -8, 0, -5, 0],
          rotateX: [0, 3, 0, -2, 0],
          rotateY: [0, 5, 0, -3, 0],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      >
        {/* Realistic Drone Body */}
        <motion.div 
          className="absolute inset-0"
          style={{
            transformStyle: 'preserve-3d',
          }}
        >
          {/* Main Central Body/Tank */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-12 h-8 md:w-16 md:h-10">
            {/* Primary Body Container */}
            <div 
              className="w-full h-full rounded-lg relative"
              style={{
                background: 'linear-gradient(135deg, #f8f9fa, #e9ecef, #dee2e6)',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3), inset 0 1px 3px rgba(255, 255, 255, 0.8)',
                border: '1px solid #adb5bd',
              }}
            >
              {/* Tank/Spray Container */}
              <div 
                className="absolute inset-1 rounded"
                style={{
                  background: 'linear-gradient(145deg, #ffffff, #f1f3f4)',
                  border: '1px solid #ced4da',
                }}
              >
                {/* Liquid Level Indicator */}
                <div 
                  className="absolute bottom-0 left-0 right-0 h-2/3 rounded-b"
                  style={{
                    background: 'linear-gradient(to top, rgba(34, 139, 34, 0.6), rgba(50, 205, 50, 0.4))',
                  }}
                />
              </div>
              
              {/* Control Module */}
              <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-4 h-2 bg-gray-800 rounded-sm">
                <motion.div
                  className="w-1 h-1 bg-[var(--drone-green)] rounded-full absolute top-0.5 left-1"
                  animate={{
                    opacity: [0.3, 1, 0.3],
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                  }}
                />
                <motion.div
                  className="w-1 h-1 bg-red-500 rounded-full absolute top-0.5 right-1"
                  animate={{
                    opacity: [1, 0.3, 1],
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                  }}
                />
              </div>

              {/* Landing Gear */}
              <div className="absolute -bottom-2 left-2 w-1 h-3 bg-gray-700 rounded-sm"></div>
              <div className="absolute -bottom-2 right-2 w-1 h-3 bg-gray-700 rounded-sm"></div>
            </div>
          </div>

          {/* Realistic Arms with Carbon Fiber Look */}
          {[
            { rotation: 45, position: 'top-1 left-1', id: 'arm1' },
            { rotation: -45, position: 'top-1 right-1', id: 'arm2' },
            { rotation: 135, position: 'bottom-1 left-1', id: 'arm3' },
            { rotation: -135, position: 'bottom-1 right-1', id: 'arm4' },
          ].map((arm, index) => (
            <motion.div
              key={arm.id}
              className={`absolute ${arm.position} w-6 h-1.5 md:w-8 md:h-2 origin-center`}
              style={{
                rotate: arm.rotation,
                background: 'linear-gradient(90deg, #2c3e50, #34495e, #2c3e50)',
                borderRadius: '2px',
                boxShadow: '0 1px 3px rgba(0, 0, 0, 0.3)',
              }}
              animate={{
                scaleY: [1, 1.1, 1],
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                delay: index * 0.3,
              }}
            />
          ))}

          {/* Realistic Motors and Propellers */}
          {[
            { position: 'top-0 left-0', direction: 1 },
            { position: 'top-0 right-0', direction: -1 },
            { position: 'bottom-0 left-0', direction: -1 },
            { position: 'bottom-0 right-0', direction: 1 },
          ].map((motor, index) => (
            <div key={`motor-${index}`} className={`absolute ${motor.position} w-6 h-6 md:w-8 md:h-8`}>
              {/* Motor Housing */}
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-4 h-3 md:w-5 md:h-4 rounded-full bg-gradient-to-b from-gray-800 to-gray-900 border border-gray-600">
                {/* Motor Center */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-gray-700 border border-gray-500"></div>
              </div>
              
              {/* Red Propeller Guards */}
              <motion.div
                className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-6 h-6 md:w-8 md:h-8 rounded-full border-2 border-red-500"
                style={{
                  background: 'radial-gradient(circle, transparent 60%, rgba(220, 38, 127, 0.1) 70%)',
                }}
                animate={{
                  scale: [1, 1.05, 1],
                  opacity: [0.8, 1, 0.8],
                }}
                transition={{
                  duration: 3,
                  repeat: Infinity,
                  delay: index * 0.2,
                }}
              />
              
              {/* Spinning Propeller Blades */}
              <motion.div
                className="absolute inset-0"
                animate={{
                  rotate: motor.direction * 360,
                }}
                transition={{
                  duration: 0.05,
                  repeat: Infinity,
                  ease: 'linear',
                }}
              >
                {/* Propeller Blade 1 */}
                <div className="absolute top-1/2 left-1/2 w-6 h-0.5 md:w-8 md:h-1 origin-left transform -translate-y-1/2 -translate-x-1/2">
                  <div 
                    className="w-full h-full"
                    style={{
                      background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.9), transparent)',
                      borderRadius: '2px',
                      boxShadow: '0 0 8px rgba(255, 255, 255, 0.6)',
                    }}
                  />
                </div>
                
                {/* Propeller Blade 2 */}
                <div className="absolute top-1/2 left-1/2 w-6 h-0.5 md:w-8 md:h-1 origin-left transform -translate-y-1/2 -translate-x-1/2 rotate-90">
                  <div 
                    className="w-full h-full"
                    style={{
                      background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.9), transparent)',
                      borderRadius: '2px',
                      boxShadow: '0 0 8px rgba(255, 255, 255, 0.6)',
                    }}
                  />
                </div>
              </motion.div>

              {/* Propeller Wash/Downwash Effect */}
              <motion.div
                className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-8 md:w-12 md:h-12 rounded-full pointer-events-none"
                style={{
                  background: 'radial-gradient(circle, rgba(255, 255, 255, 0.1), transparent 70%)',
                }}
                animate={{
                  scale: [1, 1.4, 1],
                  opacity: [0.2, 0.5, 0.2],
                }}
                transition={{
                  duration: 0.3,
                  repeat: Infinity,
                }}
              />
            </div>
          ))}

          {/* Realistic Spray System */}
          <motion.div
            className="absolute top-full left-1/2 transform -translate-x-1/2 w-4 h-8 md:w-6 md:h-12 pointer-events-none"
            animate={{ 
              opacity: [0, 0.7, 0],
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              delay: 1,
            }}
          >
            {/* Main Spray Stream */}
            <div 
              className="w-full h-full"
              style={{
                background: 'linear-gradient(to bottom, rgba(34, 139, 34, 0.6), rgba(50, 205, 50, 0.4), transparent)',
                filter: 'blur(1px)',
              }}
            />
            
            {/* Spray Nozzles */}
            <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 flex space-x-1">
              <div className="w-1 h-1 bg-gray-600 rounded-full"></div>
              <div className="w-1 h-1 bg-gray-600 rounded-full"></div>
            </div>
            
            {/* Spray Particles */}
            {[...Array(12)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-0.5 h-0.5 bg-green-400 rounded-full"
                style={{
                  left: `${30 + Math.random() * 40}%`,
                  top: `${5 + i * 7}%`,
                }}
                animate={{
                  x: [0, Math.random() * 15 - 7.5],
                  y: [0, 15 + Math.random() * 10],
                  opacity: [0.8, 0],
                  scale: [1, 0.3],
                }}
                transition={{
                  duration: 1.5 + Math.random() * 0.8,
                  repeat: Infinity,
                  delay: Math.random() * 2,
                }}
              />
            ))}
          </motion.div>

          {/* GPS/Camera Gimbal */}
          <motion.div
            className="absolute -top-3 left-1/2 transform -translate-x-1/2 w-3 h-2 bg-gray-800 rounded"
            animate={{
              rotateY: [0, 30, -30, 0],
            }}
            transition={{
              duration: 6,
              repeat: Infinity,
            }}
          >
            <div className="w-full h-full bg-gradient-to-b from-gray-700 to-gray-900 rounded flex items-center justify-center">
              <motion.div
                className="w-1 h-1 bg-blue-400 rounded-full"
                animate={{
                  opacity: [0.5, 1, 0.5],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                }}
              />
            </div>
          </motion.div>

          {/* Realistic Shadow */}
          <motion.div
            className="absolute top-full left-1/2 transform -translate-x-1/2 w-16 h-4 md:w-20 md:h-6 rounded-full opacity-30"
            style={{
              background: 'radial-gradient(ellipse, rgba(0, 0, 0, 0.4), transparent 70%)',
              filter: 'blur(3px)',
            }}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.2, 0.4, 0.2],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
            }}
          />

          {/* Flight Status Indicator */}
          <motion.div
            className="absolute -top-6 left-1/2 transform -translate-x-1/2 px-2 py-1 glass rounded text-xs text-[var(--drone-green)] border border-[var(--drone-green)]/30"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ 
              opacity: [0, 1, 1, 0],
              scale: [0.8, 1, 1, 0.8],
            }}
            transition={{
              duration: 6,
              repeat: Infinity,
              delay: 2,
            }}
          >
            <motion.span
              animate={{
                opacity: [0.7, 1, 0.7],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
              }}
            >
              SPRAYING...
            </motion.span>
          </motion.div>
        </motion.div>
      </motion.div>
    </motion.div>
  );
}