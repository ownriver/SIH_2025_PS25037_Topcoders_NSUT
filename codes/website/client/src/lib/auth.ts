import { apiRequest } from "./queryClient";

export interface AuthUser {
  id: string;
  username: string;
  email: string;
  fullName: string;
  role: string;
}

export interface AuthResponse {
  user: AuthUser;
  token: string;
}

class AuthService {
  private token: string | null = null;
  private user: AuthUser | null = null;

  constructor() {
    this.token = localStorage.getItem('auth_token');
    const userData = localStorage.getItem('auth_user');
    if (userData) {
      try {
        this.user = JSON.parse(userData);
      } catch {
        localStorage.removeItem('auth_user');
      }
    }
  }

  async login(credentials: { username: string; password: string }): Promise<AuthResponse> {
    const response = await apiRequest('POST', '/api/auth/login', credentials);
    const data: AuthResponse = await response.json();
    
    this.setAuthData(data);
    return data;
  }

  async signup(userData: { 
    username: string; 
    email: string; 
    password: string; 
    fullName: string 
  }): Promise<AuthResponse> {
    const response = await apiRequest('POST', '/api/auth/signup', userData);
    const data: AuthResponse = await response.json();
    
    this.setAuthData(data);
    return data;
  }

  async getCurrentUser(): Promise<AuthUser | null> {
    if (!this.token) return null;
    
    try {
      const response = await fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${this.token}`,
        },
      });
      
      if (!response.ok) {
        this.logout();
        return null;
      }
      
      const data = await response.json();
      this.user = data.user;
      localStorage.setItem('auth_user', JSON.stringify(this.user));
      return this.user;
    } catch {
      this.logout();
      return null;
    }
  }

  logout(): void {
    this.token = null;
    this.user = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
  }

  getToken(): string | null {
    return this.token;
  }

  getUser(): AuthUser | null {
    return this.user;
  }

  isAuthenticated(): boolean {
    return !!this.token && !!this.user;
  }

  private setAuthData(data: AuthResponse): void {
    this.token = data.token;
    this.user = data.user;
    localStorage.setItem('auth_token', this.token);
    localStorage.setItem('auth_user', JSON.stringify(this.user));
  }
}

export const authService = new AuthService();

// Helper function to add auth header to requests
export function withAuth(headers: HeadersInit = {}): HeadersInit {
  const token = authService.getToken();
  if (token) {
    return {
      ...headers,
      'Authorization': `Bearer ${token}`,
    };
  }
  return headers;
}
