import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { withAuth } from "@/lib/auth";
import { useToast } from "@/hooks/use-toast";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

import { Wifi, Bluetooth, Battery, Signal, Plus, Power, Settings, RefreshCw } from "lucide-react";

interface DroneConnection {
  id: string;
  droneName: string;
  connectionType: string;
  status: string;
  batteryLevel: number;
  lastSeen: string;
}

const droneSchema = z.object({
  droneName: z.string().min(1, "Drone name is required"),
  connectionType: z.enum(["wifi", "bluetooth"], {
    required_error: "Connection type is required",
  }),
});

type DroneFormData = z.infer<typeof droneSchema>;

export default function DroneConnectionPage() {
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [discoveredDrones, setDiscoveredDrones] = useState<Array<{id: string, name: string, type: string}>>([]);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: drones, isLoading } = useQuery({
    queryKey: ['/api/drones'],
    queryFn: async () => {
      const response = await fetch('/api/drones', {
        headers: withAuth(),
      });
      if (!response.ok) throw new Error('Failed to fetch drones');
      return response.json() as Promise<DroneConnection[]>;
    },
  });

  const createDroneMutation = useMutation({
    mutationFn: async (droneData: DroneFormData) => {
      const response = await fetch('/api/drones', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...withAuth(),
        },
        body: JSON.stringify({ ...droneData, status: 'connected', batteryLevel: 100 }),
      });
      if (!response.ok) throw new Error('Failed to connect drone');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/drones'] });
      setIsAddDialogOpen(false);
      reset();
      toast({
        title: "Drone connected successfully",
        description: "Your drone is now ready for field operations.",
      });
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Failed to connect drone",
        description: error.message,
      });
    },
  });

  const updateDroneMutation = useMutation({
    mutationFn: async ({ id, ...droneData }: { id: string } & Partial<DroneConnection>) => {
      const response = await fetch(`/api/drones/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...withAuth(),
        },
        body: JSON.stringify(droneData),
      });
      if (!response.ok) throw new Error('Failed to update drone');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/drones'] });
      toast({
        title: "Drone status updated",
        description: "Drone connection status has been updated.",
      });
    },
  });

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    setValue,
  } = useForm<DroneFormData>({
    resolver: zodResolver(droneSchema),
  });

  const onSubmit = (data: DroneFormData) => {
    createDroneMutation.mutate(data);
  };

  const handleStatusToggle = (drone: DroneConnection) => {
    const newStatus = drone.status === 'connected' ? 'disconnected' : 'connected';
    updateDroneMutation.mutate({ 
      id: drone.id, 
      status: newStatus,
      batteryLevel: newStatus === 'connected' ? Math.floor(Math.random() * 40) + 60 : drone.batteryLevel
    });
  };

  const handleStartScan = async () => {
    setIsScanning(true);
    setDiscoveredDrones([]);
    
    // Simulate drone discovery process
    setTimeout(() => {
      const mockDrones = [
        { id: 'phantom-pro', name: 'Phantom Pro', type: 'wifi' },
        { id: 'agri-bot-x1', name: 'Agri-Bot X1', type: 'bluetooth' },
        { id: 'field-scout-2', name: 'Field Scout 2', type: 'wifi' },
        { id: 'smart-sprayer', name: 'Smart Sprayer', type: 'bluetooth' }
      ];
      
      // Filter out already connected drones
      const existingNames = drones?.map(d => d.droneName.toLowerCase()) || [];
      const newDrones = mockDrones.filter(drone => 
        !existingNames.includes(drone.name.toLowerCase())
      );
      
      setDiscoveredDrones(newDrones);
      setIsScanning(false);
      
      toast({
        title: "Drone discovery completed",
        description: `Found ${newDrones.length} available drones in range.`,
      });
    }, 4000);
  };

  const handleConnectDiscoveredDrone = (discoveredDrone: {id: string, name: string, type: string}) => {
    createDroneMutation.mutate({
      droneName: discoveredDrone.name,
      connectionType: discoveredDrone.type as "wifi" | "bluetooth"
    });
    
    // Remove from discovered list
    setDiscoveredDrones(prev => prev.filter(d => d.id !== discoveredDrone.id));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'bg-primary text-primary-foreground';
      case 'scanning':
        return 'bg-secondary text-secondary-foreground';
      case 'disconnected':
        return 'bg-muted text-muted-foreground';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const getBatteryColor = (level: number) => {
    if (level > 60) return 'bg-primary';
    if (level > 30) return 'bg-accent';
    return 'bg-destructive';
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-muted rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-40 bg-muted rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Drone Connection</h1>
          <p className="text-muted-foreground">
            Manage drone connections and monitor their status for field operations
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <Button 
            variant="outline" 
            onClick={handleStartScan}
            disabled={isScanning || !drones?.some(d => d.status === 'connected')}
            data-testid="button-start-scan"
          >
            {isScanning ? (
              <RefreshCw className="mr-2 animate-spin" size={16} />
            ) : (
              <Signal className="mr-2" size={16} />
            )}
            {isScanning ? 'Scanning...' : 'Start Field Scan'}
          </Button>
          
          <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
            <DialogTrigger asChild>
              <Button data-testid="button-connect-drone">
                <Plus className="mr-2" size={16} />
                Connect Drone
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle>Connect New Drone</DialogTitle>
              </DialogHeader>
              <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="droneName">Drone Name</Label>
                  <Input
                    id="droneName"
                    placeholder="e.g., Drone Alpha-3"
                    {...register("droneName")}
                    data-testid="input-drone-name"
                  />
                  {errors.droneName && (
                    <p className="text-destructive text-sm">{errors.droneName.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="connectionType">Connection Type</Label>
                  <Select onValueChange={(value: "wifi" | "bluetooth") => setValue("connectionType", value)}>
                    <SelectTrigger data-testid="select-connection-type">
                      <SelectValue placeholder="Select connection type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="wifi">WiFi Connection</SelectItem>
                      <SelectItem value="bluetooth">Bluetooth Connection</SelectItem>
                    </SelectContent>
                  </Select>
                  {errors.connectionType && (
                    <p className="text-destructive text-sm">{errors.connectionType.message}</p>
                  )}
                </div>

                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    className="flex-1"
                    onClick={() => setIsAddDialogOpen(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    className="flex-1"
                    disabled={createDroneMutation.isPending}
                    data-testid="button-save-drone"
                  >
                    {createDroneMutation.isPending ? "Connecting..." : "Connect"}
                  </Button>
                </div>
              </form>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Connection Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                <Wifi className="text-primary" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {drones?.filter(d => d.status === 'connected').length || 0}
                </p>
                <p className="text-sm text-muted-foreground">Active Connections</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-secondary/10 rounded-lg flex items-center justify-center">
                <Battery className="text-secondary" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {drones?.length ? Math.round(drones.reduce((sum, d) => sum + d.batteryLevel, 0) / drones.length) : 0}%
                </p>
                <p className="text-sm text-muted-foreground">Average Battery</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center">
                <Signal className="text-accent" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {drones?.filter(d => d.status === 'scanning').length || 0}
                </p>
                <p className="text-sm text-muted-foreground">Currently Scanning</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Discovered Drones */}
      {discoveredDrones.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Signal className="text-primary" size={20} />
              <span>Discovered Drones</span>
              <Badge variant="secondary">{discoveredDrones.length} found</Badge>
            </CardTitle>
            <p className="text-sm text-muted-foreground">
              Select a drone to connect it to your system
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {discoveredDrones.map((discoveredDrone) => (
                <div
                  key={discoveredDrone.id}
                  className="border rounded-lg p-4 hover:bg-accent/50 transition-colors"
                  data-testid={`discovered-${discoveredDrone.id}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-muted rounded-lg flex items-center justify-center">
                        {discoveredDrone.type === 'wifi' ? (
                          <Wifi size={20} className="text-muted-foreground" />
                        ) : (
                          <Bluetooth size={20} className="text-muted-foreground" />
                        )}
                      </div>
                      <div>
                        <h4 className="font-medium">{discoveredDrone.name}</h4>
                        <p className="text-sm text-muted-foreground capitalize">
                          {discoveredDrone.type} connection
                        </p>
                      </div>
                    </div>
                    <Button 
                      size="sm"
                      onClick={() => handleConnectDiscoveredDrone(discoveredDrone)}
                      disabled={createDroneMutation.isPending}
                      data-testid={`connect-${discoveredDrone.id}`}
                    >
                      {createDroneMutation.isPending ? "Connecting..." : "Connect"}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Drone Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {drones?.map((drone) => (
          <Card key={drone.id} className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-muted rounded-lg flex items-center justify-center">
                    {drone.connectionType === 'wifi' ? (
                      <Wifi size={20} className={drone.status === 'connected' ? 'text-primary' : 'text-muted-foreground'} />
                    ) : (
                      <Bluetooth size={20} className={drone.status === 'connected' ? 'text-primary' : 'text-muted-foreground'} />
                    )}
                  </div>
                  <div>
                    <h3 className="font-medium">{drone.droneName}</h3>
                    <p className="text-sm text-muted-foreground capitalize">
                      {drone.connectionType} connection
                    </p>
                  </div>
                </div>
                
                <Badge className={getStatusColor(drone.status)} data-testid={`status-${drone.id}`}>
                  {drone.status.charAt(0).toUpperCase() + drone.status.slice(1)}
                </Badge>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-4">
              {/* Battery Level */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium flex items-center">
                    <Battery size={14} className="mr-1" />
                    Battery Level
                  </span>
                  <span className="text-sm font-bold">{drone.batteryLevel}%</span>
                </div>
                <Progress 
                  value={drone.batteryLevel} 
                  className="h-2" 
                  data-testid={`battery-${drone.id}`}
                />
                <div className={`h-1 rounded-full ${getBatteryColor(drone.batteryLevel)}`} 
                     style={{ width: `${drone.batteryLevel}%` }}></div>
              </div>

              {/* Last Seen */}
              <div className="text-sm">
                <span className="text-muted-foreground">Last seen: </span>
                <span className="font-medium">
                  {new Date(drone.lastSeen).toLocaleString()}
                </span>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2 pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={() => handleStatusToggle(drone)}
                  disabled={updateDroneMutation.isPending}
                  data-testid={`button-toggle-${drone.id}`}
                >
                  <Power size={14} className="mr-1" />
                  {drone.status === 'connected' ? 'Disconnect' : 'Connect'}
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  data-testid={`button-settings-${drone.id}`}
                >
                  <Settings size={14} />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {drones?.length === 0 && (
        <Card>
          <CardContent className="p-12 text-center">
            <Wifi className="mx-auto mb-4 text-muted-foreground" size={48} />
            <h3 className="text-lg font-medium mb-2">No drones connected</h3>
            <p className="text-muted-foreground mb-4">
              Connect your first drone to start monitoring your fields and applying precise treatments.
            </p>
            <Button onClick={() => setIsAddDialogOpen(true)} data-testid="button-connect-first-drone">
              <Plus className="mr-2" size={16} />
              Connect Your First Drone
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      {drones && drones.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Button 
                variant="outline" 
                className="h-20 flex-col space-y-2"
                onClick={() => queryClient.invalidateQueries({ queryKey: ['/api/drones'] })}
                data-testid="button-refresh-status"
              >
                <RefreshCw size={20} />
                <span>Refresh Status</span>
              </Button>
              
              <Button 
                variant="outline" 
                className="h-20 flex-col space-y-2"
                disabled={!drones.some(d => d.status === 'connected')}
                data-testid="button-sync-all"
              >
                <Wifi size={20} />
                <span>Sync All Drones</span>
              </Button>
              
              <Button 
                variant="outline" 
                className="h-20 flex-col space-y-2"
                data-testid="button-diagnostic"
              >
                <Settings size={20} />
                <span>Run Diagnostics</span>
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
