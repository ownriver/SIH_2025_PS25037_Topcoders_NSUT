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

import { Wheat, Sprout, Flower, Popcorn, Plus, Calendar, MapPin, Settings } from "lucide-react";

interface Crop {
  id: string;
  name: string;
  type: string;
  plantedDate: string;
  expectedHarvestDate: string;
  growthStage: string;
  area: number;
  isActive: boolean;
}

const cropTypes = [
  { value: "rice", label: "Rice", icon: Wheat, color: "bg-primary/10 text-primary border-primary" },
  { value: "wheat", label: "Wheat", icon: Wheat, color: "bg-secondary/10 text-secondary border-secondary" },
  { value: "cotton", label: "Cotton", icon: Flower, color: "bg-accent/10 text-accent border-accent" },
  { value: "corn", label: "Popcorn", icon: Popcorn, color: "bg-destructive/10 text-destructive border-destructive" },
];

const growthStages = ["seedling", "vegetative", "flowering", "fruiting", "maturity"];

const cropSchema = z.object({
  name: z.string().min(1, "Crop name is required"),
  type: z.string().min(1, "Crop type is required"),
  plantedDate: z.string().min(1, "Planted date is required"),
  expectedHarvestDate: z.string().min(1, "Expected harvest date is required"),
  growthStage: z.string().min(1, "Growth stage is required"),
  area: z.string().min(1, "Area is required").transform((val) => parseFloat(val)),
});

type CropFormData = z.infer<typeof cropSchema>;

export default function CropSelectionPage() {
  const [selectedCrop, setSelectedCrop] = useState<Crop | null>(null);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: crops, isLoading } = useQuery({
    queryKey: ['/api/crops'],
    queryFn: async () => {
      const response = await fetch('/api/crops', {
        headers: withAuth(),
      });
      if (!response.ok) throw new Error('Failed to fetch crops');
      return response.json() as Promise<Crop[]>;
    },
  });

  const createCropMutation = useMutation({
    mutationFn: async (cropData: CropFormData) => {
      const response = await fetch('/api/crops', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...withAuth(),
        },
        body: JSON.stringify(cropData),
      });
      if (!response.ok) throw new Error('Failed to create crop');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/crops'] });
      setIsAddDialogOpen(false);
      reset();
      toast({
        title: "Crop added successfully",
        description: "Your new crop has been added to your farm.",
      });
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Failed to add crop",
        description: error.message,
      });
    },
  });

  const updateCropMutation = useMutation({
    mutationFn: async ({ id, ...cropData }: { id: string } & Partial<Crop>) => {
      const response = await fetch(`/api/crops/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...withAuth(),
        },
        body: JSON.stringify(cropData),
      });
      if (!response.ok) throw new Error('Failed to update crop');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/crops'] });
      toast({
        title: "Crop updated successfully",
        description: "Crop information has been updated.",
      });
    },
  });

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    setValue,
    watch
  } = useForm<CropFormData>({
    resolver: zodResolver(cropSchema),
  });

  const watchedType = watch("type");

  const onSubmit = (data: CropFormData) => {
    createCropMutation.mutate(data);
  };

  const handleActivateCrop = (crop: Crop) => {
    // Deactivate all crops first, then activate selected one
    crops?.forEach((c) => {
      if (c.id !== crop.id && c.isActive) {
        updateCropMutation.mutate({ id: c.id, isActive: false });
      }
    });
    
    updateCropMutation.mutate({ id: crop.id, isActive: true });
    setSelectedCrop(crop);
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-muted rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-32 bg-muted rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const activeCrop = crops?.find(crop => crop.isActive);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Crop Selection</h1>
          <p className="text-muted-foreground">
            Manage your crops and select which one to monitor actively
          </p>
        </div>
        
        <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-add-crop">
              <Plus className="mr-2" size={16} />
              Add New Crop
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Add New Crop</DialogTitle>
            </DialogHeader>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Crop Name</Label>
                <Input
                  id="name"
                  placeholder="e.g., North Field Rice"
                  {...register("name")}
                  data-testid="input-crop-name"
                />
                {errors.name && (
                  <p className="text-destructive text-sm">{errors.name.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="type">Crop Type</Label>
                <Select onValueChange={(value) => setValue("type", value)}>
                  <SelectTrigger data-testid="select-crop-type">
                    <SelectValue placeholder="Select crop type" />
                  </SelectTrigger>
                  <SelectContent>
                    {cropTypes.map((type) => (
                      <SelectItem key={type.value} value={type.value}>
                        {type.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {errors.type && (
                  <p className="text-destructive text-sm">{errors.type.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="area">Area (hectares)</Label>
                <Input
                  id="area"
                  type="number"
                  step="0.1"
                  placeholder="e.g., 2.5"
                  {...register("area")}
                  data-testid="input-crop-area"
                />
                {errors.area && (
                  <p className="text-destructive text-sm">{errors.area.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="plantedDate">Planted Date</Label>
                <Input
                  id="plantedDate"
                  type="date"
                  {...register("plantedDate")}
                  data-testid="input-planted-date"
                />
                {errors.plantedDate && (
                  <p className="text-destructive text-sm">{errors.plantedDate.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="expectedHarvestDate">Expected Harvest Date</Label>
                <Input
                  id="expectedHarvestDate"
                  type="date"
                  {...register("expectedHarvestDate")}
                  data-testid="input-harvest-date"
                />
                {errors.expectedHarvestDate && (
                  <p className="text-destructive text-sm">{errors.expectedHarvestDate.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="growthStage">Growth Stage</Label>
                <Select onValueChange={(value) => setValue("growthStage", value)}>
                  <SelectTrigger data-testid="select-growth-stage">
                    <SelectValue placeholder="Select growth stage" />
                  </SelectTrigger>
                  <SelectContent>
                    {growthStages.map((stage) => (
                      <SelectItem key={stage} value={stage}>
                        {stage.charAt(0).toUpperCase() + stage.slice(1)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {errors.growthStage && (
                  <p className="text-destructive text-sm">{errors.growthStage.message}</p>
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
                  disabled={createCropMutation.isPending}
                  data-testid="button-save-crop"
                >
                  {createCropMutation.isPending ? "Adding..." : "Add Crop"}
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Active Crop Summary */}
      {activeCrop && (
        <Card className="border-primary/50">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center space-x-2">
                <span>Currently Active Crop</span>
                <Badge className="bg-primary text-primary-foreground">Active</Badge>
              </CardTitle>
              <Button variant="outline" size="sm">
                <Settings size={16} className="mr-2" />
                Manage
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <h4 className="font-medium mb-2">{activeCrop.name}</h4>
                <p className="text-sm text-muted-foreground">
                  {cropTypes.find(t => t.value === activeCrop.type)?.label} • {activeCrop.area} hectares
                </p>
              </div>
              <div>
                <div className="flex items-center space-x-2 mb-1">
                  <Calendar size={14} />
                  <span className="text-sm font-medium">Growth Stage</span>
                </div>
                <p className="text-sm text-primary font-medium capitalize">
                  {activeCrop.growthStage}
                </p>
              </div>
              <div>
                <div className="flex items-center space-x-2 mb-1">
                  <MapPin size={14} />
                  <span className="text-sm font-medium">Expected Harvest</span>
                </div>
                <p className="text-sm">
                  {new Date(activeCrop.expectedHarvestDate).toLocaleDateString()}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Crop Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {crops?.map((crop) => {
          const cropType = cropTypes.find(t => t.value === crop.type);
          const Icon = cropType?.icon || Wheat;
          
          return (
            <Card
              key={crop.id}
              className={`cursor-pointer transition-all hover:shadow-md ${
                crop.isActive 
                  ? 'border-primary/50 bg-primary/5' 
                  : 'hover:border-border'
              }`}
              onClick={() => !crop.isActive && handleActivateCrop(crop)}
              data-testid={`crop-card-${crop.id}`}
            >
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                      crop.isActive ? 'bg-primary text-primary-foreground' : cropType?.color
                    }`}>
                      <Icon size={24} />
                    </div>
                    <div>
                      <h3 className="font-medium">{crop.name}</h3>
                      <p className="text-sm text-muted-foreground">
                        {cropType?.label}
                      </p>
                    </div>
                  </div>
                  {crop.isActive && (
                    <Badge className="bg-primary text-primary-foreground text-xs">
                      Active
                    </Badge>
                  )}
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Area</span>
                    <span className="font-medium">{crop.area} ha</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Growth Stage</span>
                    <span className="font-medium capitalize text-primary">
                      {crop.growthStage}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Planted</span>
                    <span className="font-medium">
                      {new Date(crop.plantedDate).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Harvest Expected</span>
                    <span className="font-medium">
                      {new Date(crop.expectedHarvestDate).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                {!crop.isActive && (
                  <div className="mt-4 pt-4 border-t border-border">
                    <Button
                      size="sm"
                      variant="outline"
                      className="w-full"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleActivateCrop(crop);
                      }}
                      data-testid={`button-activate-${crop.id}`}
                    >
                      Select This Crop
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {crops?.length === 0 && (
        <Card>
          <CardContent className="p-12 text-center">
            <Sprout className="mx-auto mb-4 text-muted-foreground" size={48} />
            <h3 className="text-lg font-medium mb-2">No crops added yet</h3>
            <p className="text-muted-foreground mb-4">
              Get started by adding your first crop to begin monitoring plant health.
            </p>
            <Button onClick={() => setIsAddDialogOpen(true)} data-testid="button-add-first-crop">
              <Plus className="mr-2" size={16} />
              Add Your First Crop
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
