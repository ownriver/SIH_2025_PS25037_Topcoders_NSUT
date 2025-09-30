import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { authService, type AuthUser, type AuthResponse } from "@/lib/auth";
import { useToast } from "@/hooks/use-toast";

export function useAuth() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const { data: user, isLoading } = useQuery({
    queryKey: ['auth', 'me'],
    queryFn: () => authService.getCurrentUser(),
    staleTime: Infinity,
  });

  const loginMutation = useMutation({
    mutationFn: (credentials: { username: string; password: string }) =>
      authService.login(credentials),
    onSuccess: (data: AuthResponse) => {
      queryClient.setQueryData(['auth', 'me'], data.user);
      toast({
        title: "Login successful",
        description: `Welcome back, ${data.user.fullName}!`,
      });
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Login failed",
        description: error.message,
      });
    },
  });

  const signupMutation = useMutation({
    mutationFn: (userData: {
      username: string;
      email: string;
      password: string;
      fullName: string;
    }) => authService.signup(userData),
    onSuccess: (data: AuthResponse) => {
      queryClient.setQueryData(['auth', 'me'], data.user);
      toast({
        title: "Account created successfully",
        description: `Welcome to AgriSmart, ${data.user.fullName}!`,
      });
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Signup failed",
        description: error.message,
      });
    },
  });

  const logout = () => {
    authService.logout();
    queryClient.setQueryData(['auth', 'me'], null);
    queryClient.clear();
    toast({
      title: "Logged out",
      description: "You have been logged out successfully.",
    });
  };

  return {
    user,
    isLoading,
    isAuthenticated: !!user,
    login: loginMutation.mutate,
    signup: signupMutation.mutate,
    logout,
    isLoginPending: loginMutation.isPending,
    isSignupPending: signupMutation.isPending,
  };
}
