import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { z } from "zod";
import { useToast } from "@/hooks/use-toast";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

import { 
  Mail, 
  Phone, 
  MapPin, 
  Clock, 
  Send,
  MessageSquare,
  HeadphonesIcon,
  FileText
} from "lucide-react";

const contactSchema = z.object({
  name: z.string().min(2, "Name must be at least 2 characters"),
  email: z.string().email("Please enter a valid email address"),
  subject: z.string().min(5, "Subject must be at least 5 characters"),
  message: z.string().min(10, "Message must be at least 10 characters"),
  priority: z.enum(["low", "medium", "high"], {
    required_error: "Please select a priority level",
  }),
});

type ContactFormData = z.infer<typeof contactSchema>;

export default function ContactPage() {
  const { toast } = useToast();

  const submitContactMutation = useMutation({
    mutationFn: async (data: ContactFormData) => {
      const response = await fetch('/api/contact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error('Failed to send message');
      return response.json();
    },
    onSuccess: () => {
      reset();
      toast({
        title: "Message sent successfully",
        description: "We'll get back to you within 24 hours. Thank you for contacting AgriSmart!",
      });
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Failed to send message",
        description: error.message,
      });
    },
  });

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<ContactFormData>({
    resolver: zodResolver(contactSchema),
  });

  const onSubmit = (data: ContactFormData) => {
    submitContactMutation.mutate(data);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">Contact Support</h1>
        <p className="text-muted-foreground text-lg">
          Need help with AgriSmart? Our expert support team is here to assist you.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Contact Information */}
        <div className="space-y-6">
          {/* Support Hours */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Clock className="text-primary" size={20} />
                <span>Support Hours</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Monday - Friday</span>
                <span className="font-medium">8:00 AM - 8:00 PM</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Saturday</span>
                <span className="font-medium">9:00 AM - 5:00 PM</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Sunday</span>
                <span className="font-medium">10:00 AM - 4:00 PM</span>
              </div>
              <div className="pt-2 border-t border-border">
                <p className="text-sm text-muted-foreground">
                  Emergency support available 24/7 for critical issues
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Contact Methods */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <MessageSquare className="text-primary" size={20} />
                <span>Get In Touch</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                  <Mail className="text-primary" size={18} />
                </div>
                <div>
                  <p className="font-medium">Email Support</p>
                  <p className="text-sm text-muted-foreground">support@agrismart.com</p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-secondary/10 rounded-lg flex items-center justify-center">
                  <Phone className="text-secondary" size={18} />
                </div>
                <div>
                  <p className="font-medium">Phone Support</p>
                  <p className="text-sm text-muted-foreground">+91 800-123-4567</p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-accent/10 rounded-lg flex items-center justify-center">
                  <HeadphonesIcon className="text-accent" size={18} />
                </div>
                <div>
                  <p className="font-medium">Live Chat</p>
                  <p className="text-sm text-muted-foreground">Available during support hours</p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-muted rounded-lg flex items-center justify-center">
                  <MapPin className="text-muted-foreground" size={18} />
                </div>
                <div>
                  <p className="font-medium">Office Address</p>
                  <p className="text-sm text-muted-foreground">
                    AgriSmart Technologies<br />
                    123 Innovation Drive<br />
                    New Delhi, India 110001
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Common Issues */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <FileText className="text-primary" size={20} />
                <span>Common Issues</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <Button 
                  variant="ghost" 
                  className="w-full justify-start text-left h-auto p-3"
                  data-testid="button-faq-drone"
                >
                  <div>
                    <p className="font-medium">Drone Connection Issues</p>
                    <p className="text-sm text-muted-foreground">
                      Troubleshooting WiFi and Bluetooth connectivity
                    </p>
                  </div>
                </Button>

                <Button 
                  variant="ghost" 
                  className="w-full justify-start text-left h-auto p-3"
                  data-testid="button-faq-calibration"
                >
                  <div>
                    <p className="font-medium">Sensor Calibration</p>
                    <p className="text-sm text-muted-foreground">
                      AI detection accuracy and sensor setup
                    </p>
                  </div>
                </Button>

                <Button 
                  variant="ghost" 
                  className="w-full justify-start text-left h-auto p-3"
                  data-testid="button-faq-pesticide"
                >
                  <div>
                    <p className="font-medium">Pesticide Recommendations</p>
                    <p className="text-sm text-muted-foreground">
                      Understanding AI-powered treatment suggestions
                    </p>
                  </div>
                </Button>

                <Button 
                  variant="ghost" 
                  className="w-full justify-start text-left h-auto p-3"
                  data-testid="button-faq-billing"
                >
                  <div>
                    <p className="font-medium">Billing & Subscription</p>
                    <p className="text-sm text-muted-foreground">
                      Account management and payment issues
                    </p>
                  </div>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Contact Form */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Send className="text-primary" size={20} />
                <span>Send us a Message</span>
              </CardTitle>
              <p className="text-muted-foreground">
                Fill out the form below and we'll respond within 24 hours
              </p>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="name">Full Name *</Label>
                    <Input
                      id="name"
                      placeholder="Enter your full name"
                      {...register("name")}
                      data-testid="input-name"
                    />
                    {errors.name && (
                      <p className="text-destructive text-sm">{errors.name.message}</p>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="email">Email Address *</Label>
                    <Input
                      id="email"
                      type="email"
                      placeholder="Enter your email address"
                      {...register("email")}
                      data-testid="input-email"
                    />
                    {errors.email && (
                      <p className="text-destructive text-sm">{errors.email.message}</p>
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="subject">Subject *</Label>
                  <Input
                    id="subject"
                    placeholder="Brief description of your inquiry"
                    {...register("subject")}
                    data-testid="input-subject"
                  />
                  {errors.subject && (
                    <p className="text-destructive text-sm">{errors.subject.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="priority">Priority Level *</Label>
                  <select 
                    id="priority"
                    {...register("priority")}
                    className="w-full px-3 py-2 border border-border rounded-md bg-background"
                    data-testid="select-priority"
                  >
                    <option value="">Select priority level</option>
                    <option value="low">Low - General inquiry</option>
                    <option value="medium">Medium - Technical assistance needed</option>
                    <option value="high">High - Urgent issue affecting operations</option>
                  </select>
                  {errors.priority && (
                    <p className="text-destructive text-sm">{errors.priority.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="message">Message *</Label>
                  <Textarea
                    id="message"
                    rows={6}
                    placeholder="Please provide detailed information about your inquiry, including any error messages, steps you've taken, and your current system setup..."
                    {...register("message")}
                    data-testid="textarea-message"
                  />
                  {errors.message && (
                    <p className="text-destructive text-sm">{errors.message.message}</p>
                  )}
                  <p className="text-sm text-muted-foreground">
                    The more details you provide, the better we can assist you.
                  </p>
                </div>

                <div className="bg-muted/30 p-4 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <FileText className="text-muted-foreground mt-0.5" size={16} />
                    <div className="text-sm">
                      <p className="font-medium mb-1">For faster resolution, please include:</p>
                      <ul className="text-muted-foreground space-y-1">
                        <li>• Your account email address</li>
                        <li>• Device model and operating system</li>
                        <li>• Screenshots of any error messages</li>
                        <li>• Steps to reproduce the issue</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-4">
                  <Button
                    type="submit"
                    className="flex-1"
                    disabled={submitContactMutation.isPending}
                    data-testid="button-send-message"
                  >
                    {submitContactMutation.isPending ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground mr-2"></div>
                        Sending Message...
                      </>
                    ) : (
                      <>
                        <Send className="mr-2" size={16} />
                        Send Message
                      </>
                    )}
                  </Button>
                  
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => reset()}
                    data-testid="button-clear-form"
                  >
                    Clear Form
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>

          {/* Response Time Information */}
          <Card className="mt-6">
            <CardContent className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
                <div className="space-y-2">
                  <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
                    <Clock className="text-primary" size={20} />
                  </div>
                  <h3 className="font-medium">Response Time</h3>
                  <p className="text-sm text-muted-foreground">
                    We aim to respond to all inquiries within 24 hours
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="w-12 h-12 bg-secondary/10 rounded-full flex items-center justify-center mx-auto">
                    <HeadphonesIcon className="text-secondary" size={20} />
                  </div>
                  <h3 className="font-medium">Expert Support</h3>
                  <p className="text-sm text-muted-foreground">
                    Our team includes agricultural experts and technical specialists
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="w-12 h-12 bg-accent/10 rounded-full flex items-center justify-center mx-auto">
                    <MessageSquare className="text-accent" size={20} />
                  </div>
                  <h3 className="font-medium">Follow-up</h3>
                  <p className="text-sm text-muted-foreground">
                    We'll follow up to ensure your issue is completely resolved
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
