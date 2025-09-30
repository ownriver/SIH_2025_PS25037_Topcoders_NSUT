import { useState } from "react";
import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { 
  Leaf, 
  BarChart3, 
  Wheat, 
  Wifi, 
  Map, 
  Heart, 
  Sprout,
  Mail,
  LogOut,
  Menu,
  X
} from "lucide-react";

interface SidebarLayoutProps {
  children: React.ReactNode;
}

interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/crop-selection", label: "Crop Selection", icon: Wheat },
  { href: "/drone-connection", label: "Drone Connection", icon: Wifi },
  { href: "/field-map", label: "Field Map", icon: Map },
  { href: "/plant-health", label: "Plant Health", icon: Heart },
  { href: "/pesticide-control", label: "Pesticide Control", icon: Sprout },
];

export function SidebarLayout({ children }: SidebarLayoutProps) {
  const [location] = useLocation();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const { user, logout } = useAuth();

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  return (
    <div className="min-h-screen flex bg-background">
      {/* Mobile overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside 
        className={cn(
          "w-64 bg-sidebar border-r border-sidebar-border flex-shrink-0 z-50 lg:relative lg:translate-x-0 transition-transform duration-200",
          "fixed inset-y-0 left-0",
          isSidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
          "lg:block"
        )}
      >
        <div className="p-6">
          {/* Logo */}
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
                <Leaf className="text-primary-foreground" size={20} />
              </div>
              <div>
                <h1 className="text-lg font-bold text-sidebar-foreground">KisanBuddy</h1>
                <p className="text-xs text-muted-foreground">Agricultural Drone Monitoring</p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="lg:hidden"
              onClick={toggleSidebar}
              data-testid="button-close-sidebar"
            >
              <X size={16} />
            </Button>
          </div>
          
          {/* Navigation */}
          <nav className="space-y-2">
            {navItems.map((item) => {
              const isActive = location === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center space-x-3 px-3 py-2 rounded-md transition-colors",
                    isActive
                      ? "bg-sidebar-primary text-sidebar-primary-foreground"
                      : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                  )}
                  data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  <item.icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>
          
          {/* Bottom section */}
          <div className="mt-8 pt-6 border-t border-sidebar-border">
            <Link
              href="/contact"
              className="flex items-center space-x-3 px-3 py-2 rounded-md text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors"
              data-testid="nav-contact"
            >
              <Mail size={20} />
              <span>Contact Support</span>
            </Link>
            <Button
              variant="ghost"
              className="w-full justify-start space-x-3 px-3 py-2 mt-2 text-sidebar-foreground hover:bg-destructive hover:text-destructive-foreground transition-colors"
              onClick={logout}
              data-testid="button-logout"
            >
              <LogOut size={20} />
              <span>Logout</span>
            </Button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {/* Top header */}
        <header className="bg-card border-b border-border p-4 lg:p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                size="sm"
                className="lg:hidden"
                onClick={toggleSidebar}
                data-testid="button-open-sidebar"
              >
                <Menu size={20} />
              </Button>
              <div>
                <h2 className="text-xl font-semibold text-card-foreground">
                  {navItems.find(item => item.href === location)?.label || "KisanBuddy"}
                </h2>
                <p className="text-sm text-muted-foreground">
                  {new Date().toLocaleDateString('en-US', { 
                    weekday: 'long', 
                    year: 'numeric', 
                    month: 'long', 
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="hidden md:flex items-center space-x-2 bg-muted px-3 py-2 rounded-lg">
                <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                <span className="text-sm text-muted-foreground">Online</span>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-secondary rounded-full flex items-center justify-center">
                  <span className="text-secondary-foreground text-sm font-medium">
                    {user?.fullName?.charAt(0) || 'U'}
                  </span>
                </div>
                <div className="hidden md:block">
                  <p className="text-sm font-medium text-card-foreground" data-testid="text-user-name">
                    {user?.fullName || 'User'}
                  </p>
                  <p className="text-xs text-muted-foreground capitalize">
                    {user?.role || 'Farmer'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <div className="p-4 lg:p-6">
          {children}
        </div>
      </main>
    </div>
  );
}
