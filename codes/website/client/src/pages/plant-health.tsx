import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { withAuth } from "@/lib/auth";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";

import { 
  Heart, 
  AlertTriangle, 
  CheckCircle, 
  Activity, 
  TrendingUp, 
  TrendingDown,
  Thermometer,
  Droplets,
  Zap,
  Calendar,
  MapPin
} from "lucide-react";

interface PlantHealthRecord {
  id: string;
  fieldId: string;
  droneId?: string;
  healthScore: number;
  infectionRate: number;
  infectionType?: string;
  severity: string;
  latitude?: number;
  longitude?: number;
  detectionConfidence: number;
  recordedAt: string;
}

interface Field {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  area: number;
}

export default function PlantHealthPage() {
  const [selectedField, setSelectedField] = useState<string>("");
  const [timeRange, setTimeRange] = useState<string>("7d");
  const [severityFilter, setSeverityFilter] = useState<string>("all");

  const { data: healthRecords, isLoading } = useQuery({
    queryKey: ['/api/health-records'],
    queryFn: async () => {
      const response = await fetch('/api/health-records', {
        headers: withAuth(),
      });
      if (!response.ok) throw new Error('Failed to fetch health records');
      return response.json() as Promise<PlantHealthRecord[]>;
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

  // Filter and process health records
  const filteredRecords = healthRecords?.filter(record => {
    const matchesField = !selectedField || record.fieldId === selectedField;
    const matchesSeverity = severityFilter === 'all' || record.severity === severityFilter;
    
    // Time range filter
    const recordDate = new Date(record.recordedAt);
    const now = new Date();
    const daysAgo = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
    const cutoffDate = new Date(now.getTime() - daysAgo * 24 * 60 * 60 * 1000);
    const matchesTime = recordDate >= cutoffDate;
    
    return matchesField && matchesSeverity && matchesTime;
  }) || [];

  // Calculate statistics
  const stats = {
    totalRecords: filteredRecords.length,
    averageHealth: filteredRecords.length > 0 
      ? Math.round(filteredRecords.reduce((sum, r) => sum + r.healthScore, 0) / filteredRecords.length) 
      : 0,
    averageInfection: filteredRecords.length > 0 
      ? Math.round((filteredRecords.reduce((sum, r) => sum + r.infectionRate, 0) / filteredRecords.length) * 10) / 10 
      : 0,
    highRisk: filteredRecords.filter(r => r.severity === 'high').length,
    mediumRisk: filteredRecords.filter(r => r.severity === 'medium').length,
    lowRisk: filteredRecords.filter(r => r.severity === 'low').length,
    averageConfidence: filteredRecords.length > 0 
      ? Math.round(filteredRecords.reduce((sum, r) => sum + r.detectionConfidence, 0) / filteredRecords.length) 
      : 0,
  };

  // Group records by infection type
  const infectionTypes = filteredRecords.reduce((acc, record) => {
    if (record.infectionType) {
      acc[record.infectionType] = (acc[record.infectionType] || 0) + 1;
    }
    return acc;
  }, {} as Record<string, number>);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'text-destructive bg-destructive/10 border-destructive/20';
      case 'medium':
        return 'text-accent bg-accent/10 border-accent/20';
      case 'low':
        return 'text-primary bg-primary/10 border-primary/20';
      default:
        return 'text-muted-foreground bg-muted/10 border-border';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'high':
        return <AlertTriangle className="text-destructive" size={16} />;
      case 'medium':
        return <Activity className="text-accent" size={16} />;
      case 'low':
        return <CheckCircle className="text-primary" size={16} />;
      default:
        return <Heart className="text-muted-foreground" size={16} />;
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
          <h1 className="text-2xl font-bold">Plant Health Monitoring</h1>
          <p className="text-muted-foreground">
            Real-time plant health analysis powered by AI detection algorithms
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <Select value={selectedField} onValueChange={setSelectedField}>
            <SelectTrigger className="w-48" data-testid="select-field-filter">
              <SelectValue placeholder="All fields" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Fields</SelectItem>
              {fields?.map((field) => (
                <SelectItem key={field.id} value={field.id}>
                  {field.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32" data-testid="select-time-range">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
              <SelectItem value="90d">Last 3 months</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Health Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                <Heart className="text-primary" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold" data-testid="stat-avg-health">
                  {stats.averageHealth}%
                </p>
                <p className="text-sm text-muted-foreground">Average Health</p>
              </div>
            </div>
            <div className="mt-4">
              <Progress value={stats.averageHealth} className="h-2" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center">
                <AlertTriangle className="text-accent" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold text-accent" data-testid="stat-infection-rate">
                  {stats.averageInfection}%
                </p>
                <p className="text-sm text-muted-foreground">Infection Rate</p>
              </div>
            </div>
            <div className="flex items-center mt-4 text-sm">
              {stats.averageInfection <= 5 ? (
                <>
                  <TrendingDown className="text-primary mr-1" size={12} />
                  <span className="text-primary">Decreasing</span>
                </>
              ) : (
                <>
                  <TrendingUp className="text-destructive mr-1" size={12} />
                  <span className="text-destructive">Increasing</span>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-destructive/10 rounded-lg flex items-center justify-center">
                <Activity className="text-destructive" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold" data-testid="stat-high-risk">
                  {stats.highRisk}
                </p>
                <p className="text-sm text-muted-foreground">High Risk Areas</p>
              </div>
            </div>
            <div className="flex items-center mt-4 text-sm">
              <span className="text-muted-foreground">
                {stats.mediumRisk} medium, {stats.lowRisk} low risk
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-secondary/10 rounded-lg flex items-center justify-center">
                <Zap className="text-secondary" size={24} />
              </div>
              <div>
                <p className="text-2xl font-bold" data-testid="stat-confidence">
                  {stats.averageConfidence}%
                </p>
                <p className="text-sm text-muted-foreground">AI Confidence</p>
              </div>
            </div>
            <div className="flex items-center mt-4 text-sm">
              <CheckCircle className="text-primary mr-1" size={12} />
              <span className="text-primary">High Accuracy</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analysis */}
      <Tabs defaultValue="records" className="space-y-6">
        <TabsList>
          <TabsTrigger value="records">Health Records</TabsTrigger>
          <TabsTrigger value="infections">Infection Analysis</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
        </TabsList>

        <TabsContent value="records" className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Recent Health Records</h3>
            <Select value={severityFilter} onValueChange={setSeverityFilter}>
              <SelectTrigger className="w-40" data-testid="select-severity-filter">
                <SelectValue placeholder="All severities" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severities</SelectItem>
                <SelectItem value="high">High Risk</SelectItem>
                <SelectItem value="medium">Medium Risk</SelectItem>
                <SelectItem value="low">Low Risk</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid gap-4">
            {filteredRecords.slice(0, 10).map((record) => {
              const field = fields?.find(f => f.id === record.fieldId);
              return (
                <Card key={record.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-4">
                        <div className="flex-shrink-0">
                          {getSeverityIcon(record.severity)}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            <h4 className="font-medium">{field?.name || 'Unknown Field'}</h4>
                            <Badge className={getSeverityColor(record.severity)}>
                              {record.severity.charAt(0).toUpperCase() + record.severity.slice(1)} Risk
                            </Badge>
                          </div>
                          
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <span className="text-muted-foreground">Health Score</span>
                              <p className="font-medium text-primary">{record.healthScore}%</p>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Infection Rate</span>
                              <p className="font-medium">{record.infectionRate}%</p>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Confidence</span>
                              <p className="font-medium">{record.detectionConfidence}%</p>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Detection Time</span>
                              <p className="font-medium">
                                {new Date(record.recordedAt).toLocaleString()}
                              </p>
                            </div>
                          </div>
                          
                          {record.infectionType && (
                            <div className="mt-2">
                              <span className="text-sm text-muted-foreground">Infection Type: </span>
                              <span className="text-sm font-medium capitalize">{record.infectionType}</span>
                            </div>
                          )}
                          
                          {record.latitude && record.longitude && (
                            <div className="flex items-center mt-2 text-sm text-muted-foreground">
                              <MapPin size={14} className="mr-1" />
                              <span>
                                {record.latitude.toFixed(4)}, {record.longitude.toFixed(4)}
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Button variant="outline" size="sm" data-testid={`button-view-${record.id}`}>
                          View Details
                        </Button>
                        {record.severity === 'high' && (
                          <Button size="sm" data-testid={`button-treat-${record.id}`}>
                            Recommend Treatment
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {filteredRecords.length === 0 && (
            <Card>
              <CardContent className="p-12 text-center">
                <Heart className="mx-auto mb-4 text-muted-foreground" size={48} />
                <h3 className="text-lg font-medium mb-2">No health records found</h3>
                <p className="text-muted-foreground">
                  No health records match your current filters. Try adjusting the time range or field selection.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="infections" className="space-y-4">
          <h3 className="text-lg font-semibold">Infection Type Analysis</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Infection Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(infectionTypes).map(([type, count]) => (
                    <div key={type} className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm font-medium capitalize">{type}</span>
                        <span className="text-sm text-muted-foreground">{count} cases</span>
                      </div>
                      <Progress 
                        value={(count / Math.max(...Object.values(infectionTypes))) * 100} 
                        className="h-2"
                      />
                    </div>
                  ))}
                  
                  {Object.keys(infectionTypes).length === 0 && (
                    <p className="text-sm text-muted-foreground text-center py-8">
                      No infection data available for the selected period.
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Risk Level Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium flex items-center">
                        <div className="w-3 h-3 bg-destructive rounded-full mr-2"></div>
                        High Risk
                      </span>
                      <span className="text-sm text-muted-foreground">{stats.highRisk}</span>
                    </div>
                    <Progress value={stats.totalRecords ? (stats.highRisk / stats.totalRecords) * 100 : 0} className="h-2" />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium flex items-center">
                        <div className="w-3 h-3 bg-accent rounded-full mr-2"></div>
                        Medium Risk
                      </span>
                      <span className="text-sm text-muted-foreground">{stats.mediumRisk}</span>
                    </div>
                    <Progress value={stats.totalRecords ? (stats.mediumRisk / stats.totalRecords) * 100 : 0} className="h-2" />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium flex items-center">
                        <div className="w-3 h-3 bg-primary rounded-full mr-2"></div>
                        Low Risk
                      </span>
                      <span className="text-sm text-muted-foreground">{stats.lowRisk}</span>
                    </div>
                    <Progress value={stats.totalRecords ? (stats.lowRisk / stats.totalRecords) * 100 : 0} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="trends">
          <Card>
            <CardHeader>
              <CardTitle>Health Trends Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-3">
                    <TrendingUp className="text-primary" size={24} />
                  </div>
                  <h4 className="font-medium mb-2">Overall Health Trend</h4>
                  <p className="text-sm text-muted-foreground">
                    Plant health has improved by 12% over the selected period
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="w-16 h-16 bg-accent/10 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Thermometer className="text-accent" size={24} />
                  </div>
                  <h4 className="font-medium mb-2">Environmental Impact</h4>
                  <p className="text-sm text-muted-foreground">
                    Weather conditions have been favorable for plant growth
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="w-16 h-16 bg-secondary/10 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Droplets className="text-secondary" size={24} />
                  </div>
                  <h4 className="font-medium mb-2">Treatment Effectiveness</h4>
                  <p className="text-sm text-muted-foreground">
                    Applied treatments show 89% success rate in affected areas
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
