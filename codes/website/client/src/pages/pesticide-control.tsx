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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";

import { 
  Sprout, 
  Plus, 
  CheckCircle, 
  Clock, 
  AlertTriangle, 
  Droplets,
  Calendar,
  Target,
  BarChart3,
  TrendingUp,
  Award,
  Activity
} from "lucide-react";

interface PesticideApplication {
  id: string;
  fieldId: string;
  healthRecordId?: string;
  pesticideType: string;
  volumePerHectare: number;
  totalVolume: number;
  applicationMethod: string;
  status: string;
  recommendedBy: string;
  confidence: number;
  scheduledFor?: string;
  appliedAt?: string;
  createdAt: string;
}

interface Field {
  id: string;
  name: string;
  area: number;
}

const applicationSchema = z.object({
  fieldId: z.string().min(1, "Field selection is required"),
  pesticideType: z.string().min(1, "Pesticide type is required"),
  volumePerHectare: z.string().min(1, "Volume per hectare is required").transform((val) => parseFloat(val)),
  applicationMethod: z.string().min(1, "Application method is required"),
  scheduledFor: z.string().optional(),
  notes: z.string().optional(),
});

type ApplicationFormData = z.infer<typeof applicationSchema>;

export default function PesticideControlPage() {
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [selectedStatus, setSelectedStatus] = useState<string>("all");
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: applications, isLoading } = useQuery({
    queryKey: ['/api/pesticide-applications'],
    queryFn: async () => {
      const response = await fetch('/api/pesticide-applications', {
        headers: withAuth(),
      });
      if (!response.ok) throw new Error('Failed to fetch applications');
      return response.json() as Promise<PesticideApplication[]>;
    },
  });

  const { data: fields } = useQuery({
    queryKey: ['/api/fields'],
    queryFn: async () => {
      const response = await fetch('/api/fields', {
        headers: withAuth(),
      });
      if (!response.ok) throw new Error('Failed to fetch fields');
      return response.json() as Promise<Field[]>;
    },
  });

  const createApplicationMutation = useMutation({
    mutationFn: async (applicationData: ApplicationFormData) => {
      const field = fields?.find(f => f.id === applicationData.fieldId);
      const totalVolume = field ? applicationData.volumePerHectare * field.area : applicationData.volumePerHectare;
      
      const response = await fetch('/api/pesticide-applications', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...withAuth(),
        },
        body: JSON.stringify({
          ...applicationData,
          totalVolume,
          status: applicationData.scheduledFor ? 'scheduled' : 'recommended',
          recommendedBy: 'manual',
          confidence: 95,
        }),
      });
      if (!response.ok) throw new Error('Failed to create application');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/pesticide-applications'] });
      setIsAddDialogOpen(false);
      reset();
      toast({
        title: "Application created successfully",
        description: "The pesticide application has been added to your schedule.",
      });
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Failed to create application",
        description: error.message,
      });
    },
  });

  const updateApplicationMutation = useMutation({
    mutationFn: async ({ id, ...data }: { id: string } & Partial<PesticideApplication>) => {
      const response = await fetch(`/api/pesticide-applications/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...withAuth(),
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error('Failed to update application');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/pesticide-applications'] });
      toast({
        title: "Application updated",
        description: "The application status has been updated.",
      });
    },
  });

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    setValue,
    watch,
  } = useForm<ApplicationFormData>({
    resolver: zodResolver(applicationSchema),
  });

  const watchedFieldId = watch("fieldId");
  const watchedVolume = watch("volumePerHectare");

  const onSubmit = (data: ApplicationFormData) => {
    createApplicationMutation.mutate(data);
  };

  const handleStatusChange = (application: PesticideApplication, newStatus: string) => {
    const updateData: Partial<PesticideApplication> = { status: newStatus };
    
    if (newStatus === 'applied') {
      updateData.appliedAt = new Date().toISOString();
    }
    
    updateApplicationMutation.mutate({ id: application.id, ...updateData });
  };

  // Filter applications based on selected status
  const filteredApplications = applications?.filter(app => {
    if (selectedStatus === 'all') return true;
    return app.status === selectedStatus;
  }) || [];

  // Calculate statistics
  const stats = {
    total: applications?.length || 0,
    pending: applications?.filter(a => a.status === 'recommended' || a.status === 'scheduled').length || 0,
    completed: applications?.filter(a => a.status === 'applied' || a.status === 'completed').length || 0,
    totalVolume: applications?.reduce((sum, a) => sum + a.totalVolume, 0) || 0,
    avgConfidence: applications?.length ? 
      Math.round(applications.reduce((sum, a) => sum + a.confidence, 0) / applications.length) : 0,
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'recommended':
        return 'bg-accent text-accent-foreground';
      case 'scheduled':
        return 'bg-secondary text-secondary-foreground';
      case 'applied':
        return 'bg-primary text-primary-foreground';
      case 'completed':
        return 'bg-primary text-primary-foreground';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'recommended':
        return <AlertTriangle size={16} />;
      case 'scheduled':
        return <Clock size={16} />;
      case 'applied':
      case 'completed':
        return <CheckCircle size={16} />;
      default:
        return <Activity size={16} />;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-muted rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-24 bg-muted rounded-lg"></div>
            ))}
          </div>
          <div className="h-96 bg-muted rounded-lg"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Pesticide Control</h1>
          <p className="text-muted-foreground">
            Manage pesticide applications and monitor treatment effectiveness
          </p>
        </div>
        
        <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-add-application">
              <Plus className="mr-2" size={16} />
              Schedule Application
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Schedule Pesticide Application</DialogTitle>
            </DialogHeader>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="fieldId">Field</Label>
                <Select onValueChange={(value) => setValue("fieldId", value)}>
                  <SelectTrigger data-testid="select-field">
                    <SelectValue placeholder="Select field" />
                  </SelectTrigger>
                  <SelectContent>
                    {fields?.map((field) => (
                      <SelectItem key={field.id} value={field.id}>
                        {field.name} ({field.area} ha)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {errors.fieldId && (
                  <p className="text-destructive text-sm">{errors.fieldId.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="pesticideType">Pesticide Type</Label>
                <Select onValueChange={(value) => setValue("pesticideType", value)}>
                  <SelectTrigger data-testid="select-pesticide">
                    <SelectValue placeholder="Select pesticide" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="neem_oil">Neem Oil Spray</SelectItem>
                    <SelectItem value="organic_fungicide">Organic Fungicide</SelectItem>
                    <SelectItem value="insecticidal_soap">Insecticidal Soap</SelectItem>
                    <SelectItem value="copper_sulfate">Copper Sulfate</SelectItem>
                    <SelectItem value="bacillus_thuringiensis">Bacillus Thuringiensis</SelectItem>
                  </SelectContent>
                </Select>
                {errors.pesticideType && (
                  <p className="text-destructive text-sm">{errors.pesticideType.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="volumePerHectare">Volume per Hectare (L)</Label>
                <Input
                  id="volumePerHectare"
                  type="number"
                  step="0.1"
                  placeholder="e.g., 2.5"
                  {...register("volumePerHectare")}
                  data-testid="input-volume"
                />
                {errors.volumePerHectare && (
                  <p className="text-destructive text-sm">{errors.volumePerHectare.message}</p>
                )}
                {watchedFieldId && watchedVolume && (
                  <p className="text-sm text-muted-foreground">
                    Total volume needed: {(parseFloat(String(watchedVolume || 0)) * (fields?.find(f => f.id === watchedFieldId)?.area || 0)).toFixed(1)}L
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="applicationMethod">Application Method</Label>
                <Select onValueChange={(value) => setValue("applicationMethod", value)}>
                  <SelectTrigger data-testid="select-method">
                    <SelectValue placeholder="Select method" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="drone">Drone Application</SelectItem>
                    <SelectItem value="ground_sprayer">Ground Sprayer</SelectItem>
                    <SelectItem value="manual">Manual Application</SelectItem>
                  </SelectContent>
                </Select>
                {errors.applicationMethod && (
                  <p className="text-destructive text-sm">{errors.applicationMethod.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="scheduledFor">Schedule For (Optional)</Label>
                <Input
                  id="scheduledFor"
                  type="datetime-local"
                  {...register("scheduledFor")}
                  data-testid="input-schedule"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="notes">Notes (Optional)</Label>
                <Textarea
                  id="notes"
                  placeholder="Additional notes or instructions..."
                  {...register("notes")}
                  data-testid="textarea-notes"
                />
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
                  disabled={createApplicationMutation.isPending}
                  data-testid="button-save-application"
                >
                  {createApplicationMutation.isPending ? "Creating..." : "Create Application"}
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                <Sprout className="text-primary" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold" data-testid="stat-total-applications">
                  {stats.total}
                </p>
                <p className="text-sm text-muted-foreground">Total Applications</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center">
                <Clock className="text-accent" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold text-accent" data-testid="stat-pending">
                  {stats.pending}
                </p>
                <p className="text-sm text-muted-foreground">Pending</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-secondary/10 rounded-lg flex items-center justify-center">
                <Droplets className="text-secondary" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold" data-testid="stat-total-volume">
                  {stats.totalVolume.toFixed(1)}L
                </p>
                <p className="text-sm text-muted-foreground">Total Volume Used</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                <Award className="text-primary" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold" data-testid="stat-avg-confidence">
                  {stats.avgConfidence}%
                </p>
                <p className="text-sm text-muted-foreground">Avg Confidence</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Applications Management */}
      <Tabs defaultValue="applications" className="space-y-6">
        <TabsList>
          <TabsTrigger value="applications">Applications</TabsTrigger>
          <TabsTrigger value="recommendations">AI Recommendations</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="applications" className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Application Management</h3>
            <Select value={selectedStatus} onValueChange={setSelectedStatus}>
              <SelectTrigger className="w-48" data-testid="select-status-filter">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Applications</SelectItem>
                <SelectItem value="recommended">Recommended</SelectItem>
                <SelectItem value="scheduled">Scheduled</SelectItem>
                <SelectItem value="applied">Applied</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid gap-4">
            {filteredApplications.map((application) => {
              const field = fields?.find(f => f.id === application.fieldId);
              return (
                <Card key={application.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-4 flex-1">
                        <div className="flex-shrink-0 mt-1">
                          {getStatusIcon(application.status)}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            <h4 className="font-medium">{field?.name || 'Unknown Field'}</h4>
                            <Badge className={getStatusColor(application.status)}>
                              {application.status.charAt(0).toUpperCase() + application.status.slice(1)}
                            </Badge>
                            <Badge variant="outline">
                              {application.confidence}% confidence
                            </Badge>
                          </div>
                          
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-3">
                            <div>
                              <span className="text-muted-foreground">Pesticide</span>
                              <p className="font-medium capitalize">{application.pesticideType.replace('_', ' ')}</p>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Volume</span>
                              <p className="font-medium">{application.totalVolume.toFixed(1)}L</p>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Method</span>
                              <p className="font-medium capitalize">{application.applicationMethod.replace('_', ' ')}</p>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Created</span>
                              <p className="font-medium">
                                {new Date(application.createdAt).toLocaleDateString()}
                              </p>
                            </div>
                          </div>
                          
                          {application.scheduledFor && (
                            <div className="flex items-center text-sm text-muted-foreground mb-2">
                              <Calendar size={14} className="mr-1" />
                              <span>
                                Scheduled for: {new Date(application.scheduledFor).toLocaleString()}
                              </span>
                            </div>
                          )}
                          
                          {application.appliedAt && (
                            <div className="flex items-center text-sm text-muted-foreground">
                              <CheckCircle size={14} className="mr-1" />
                              <span>
                                Applied on: {new Date(application.appliedAt).toLocaleString()}
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        {application.status === 'recommended' && (
                          <Button
                            size="sm"
                            onClick={() => handleStatusChange(application, 'scheduled')}
                            disabled={updateApplicationMutation.isPending}
                            data-testid={`button-schedule-${application.id}`}
                          >
                            Schedule
                          </Button>
                        )}
                        
                        {application.status === 'scheduled' && (
                          <Button
                            size="sm"
                            onClick={() => handleStatusChange(application, 'applied')}
                            disabled={updateApplicationMutation.isPending}
                            data-testid={`button-apply-${application.id}`}
                          >
                            Mark Applied
                          </Button>
                        )}
                        
                        <Button variant="outline" size="sm" data-testid={`button-edit-${application.id}`}>
                          Edit
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {filteredApplications.length === 0 && (
            <Card>
              <CardContent className="p-12 text-center">
                <Sprout className="mx-auto mb-4 text-muted-foreground" size={48} />
                <h3 className="text-lg font-medium mb-2">No applications found</h3>
                <p className="text-muted-foreground mb-4">
                  {selectedStatus === 'all' 
                    ? "No pesticide applications have been created yet." 
                    : `No applications with status "${selectedStatus}" found.`}
                </p>
                <Button onClick={() => setIsAddDialogOpen(true)} data-testid="button-create-first-application">
                  <Plus className="mr-2" size={16} />
                  Create Your First Application
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="recommendations">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Target size={20} />
                <span>AI-Powered Recommendations</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Sample AI recommendations */}
                <div className="p-4 bg-accent/10 border border-accent/30 rounded-lg">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
                        <AlertTriangle className="text-accent-foreground" size={16} />
                      </div>
                      <div>
                        <h4 className="font-medium">Medium Priority Treatment</h4>
                        <p className="text-sm text-muted-foreground">North Field Zone A-3</p>
                      </div>
                    </div>
                    <Badge className="bg-accent text-accent-foreground">85% Confidence</Badge>
                  </div>
                  
                  <div className="space-y-2 text-sm mb-4">
                    <p><strong>Issue:</strong> Aphid infestation detected (15% coverage)</p>
                    <p><strong>Recommended:</strong> Neem oil spray - 2.5L per hectare</p>
                    <p><strong>Optimal timing:</strong> Early morning application (6-8 AM)</p>
                    <p><strong>Expected effectiveness:</strong> 92% reduction in 48 hours</p>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-muted-foreground">
                      Generated 15 minutes ago
                    </div>
                    <div className="space-x-2">
                      <Button variant="outline" size="sm">
                        Modify
                      </Button>
                      <Button size="sm" data-testid="button-accept-recommendation">
                        Accept & Schedule
                      </Button>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-primary/10 border border-primary/30 rounded-lg">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                        <CheckCircle className="text-primary-foreground" size={16} />
                      </div>
                      <div>
                        <h4 className="font-medium">Preventive Treatment</h4>
                        <p className="text-sm text-muted-foreground">South Field Zone B-1</p>
                      </div>
                    </div>
                    <Badge className="bg-primary text-primary-foreground">92% Confidence</Badge>
                  </div>
                  
                  <div className="space-y-2 text-sm mb-4">
                    <p><strong>Issue:</strong> Seasonal fungal prevention due</p>
                    <p><strong>Recommended:</strong> Organic fungicide - 1.8L per hectare</p>
                    <p><strong>Optimal timing:</strong> Evening application (6-8 PM)</p>
                    <p><strong>Expected benefit:</strong> 78% reduction in fungal risk</p>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-muted-foreground">
                      Generated 1 hour ago
                    </div>
                    <div className="space-x-2">
                      <Button variant="outline" size="sm">
                        Delay
                      </Button>
                      <Button size="sm" data-testid="button-schedule-preventive">
                        Schedule for Tomorrow
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 size={20} />
                  <span>Usage Analytics</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Pesticide Efficiency</span>
                      <span className="font-medium">94%</span>
                    </div>
                    <Progress value={94} className="h-2" />
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Cost Reduction</span>
                      <span className="font-medium">76%</span>
                    </div>
                    <Progress value={76} className="h-2" />
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Application Accuracy</span>
                      <span className="font-medium">89%</span>
                    </div>
                    <Progress value={89} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp size={20} />
                  <span>Effectiveness Trends</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-center p-4 bg-primary/10 rounded-lg">
                    <div className="text-2xl font-bold text-primary mb-1">247L</div>
                    <div className="text-sm text-muted-foreground">Pesticide Saved This Month</div>
                  </div>
                  
                  <div className="text-center p-4 bg-secondary/10 rounded-lg">
                    <div className="text-2xl font-bold text-secondary mb-1">₹12,450</div>
                    <div className="text-sm text-muted-foreground">Cost Savings This Month</div>
                  </div>
                  
                  <div className="text-center p-4 bg-accent/10 rounded-lg">
                    <div className="text-2xl font-bold text-accent mb-1">98.5%</div>
                    <div className="text-sm text-muted-foreground">Treatment Success Rate</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
